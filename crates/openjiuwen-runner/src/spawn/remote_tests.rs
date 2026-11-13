#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use openjiuwen_common::error::ErrorKind;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Serializes `frame` as one NDJSON line (without the trailing newline).
fn frame_line(frame: &SpawnMessage) -> String {
    frame.to_line().unwrap().trim_end().to_owned()
}

/// Serializes `chunk` as a `stream_chunk` line.
fn chunk_line(index: i64, text: &str) -> String {
    frame_line(&SpawnMessage::new(
        SpawnMessageKind::StreamChunk,
        serde_json::to_value(AgentChunk::llm_output(index, text)).unwrap(),
    ))
}

fn server_agent(server: &MockServer, agent_id: &str) -> RemoteAgent {
    RemoteAgent::new(RemoteAgentConfig::new(agent_id, server.uri()))
}

#[test]
fn config_defaults_match_the_python_layout() {
    let config = RemoteAgentConfig::default();
    assert_eq!(config.agent_id, "");
    assert_eq!(config.base_url, "");
    assert_eq!(config.invoke_path, "agents/{agent_id}/invoke");
    assert_eq!(config.stream_path, "agents/{agent_id}/stream");
    assert_eq!(
        config.cancel_path.as_deref(),
        Some("agents/{agent_id}/cancel")
    );
    assert_eq!(config.session_id, None);
    assert!((config.timeout_secs - 60.0).abs() < f64::EPSILON);
    assert!(config.headers.is_empty());
    assert_eq!(config.effective_timeout(), Duration::from_secs(60));
}

#[test]
fn config_round_trips_through_json() {
    let config = RemoteAgentConfig::new("assistant", "http://127.0.0.1:8080")
        .with_paths("a/{agent_id}/run", "a/{agent_id}/events")
        .with_cancel_path(Some("a/{agent_id}/stop".to_owned()))
        .with_session_id("s-1")
        .with_timeout(Duration::from_secs(5))
        .with_header("Authorization", "Bearer token");
    let value = serde_json::to_value(&config).unwrap();
    assert_eq!(value["agent_id"], json!("assistant"));
    assert_eq!(value["invoke_path"], json!("a/{agent_id}/run"));
    assert_eq!(value["cancel_path"], json!("a/{agent_id}/stop"));
    assert_eq!(value["timeout_secs"], json!(5.0));
    let restored: RemoteAgentConfig = serde_json::from_value(value).unwrap();
    assert_eq!(restored, config);
}

#[test]
fn config_tolerates_unknown_fields_and_partial_json() {
    let config: RemoteAgentConfig = serde_json::from_value(json!({
        "agent_id": "assistant",
        "base_url": "http://127.0.0.1:8080",
        "retry_policy": {"max": 3},
    }))
    .unwrap();
    assert_eq!(config.agent_id, "assistant");
    assert_eq!(config.invoke_path, DEFAULT_INVOKE_PATH);
    assert_eq!(config.stream_path, DEFAULT_STREAM_PATH);
    assert!((config.timeout_secs - 60.0).abs() < f64::EPSILON);
    assert!(config.headers.is_empty());
}

#[test]
fn cancel_path_none_round_trips() {
    let config = RemoteAgentConfig::new("a", "http://host").with_cancel_path(None);
    let restored: RemoteAgentConfig =
        serde_json::from_value(serde_json::to_value(&config).unwrap()).unwrap();
    assert_eq!(restored.cancel_path, None);
    assert_eq!(restored.cancel_url().unwrap(), None);
}

#[test]
fn effective_timeout_sanitizes_unusable_values() {
    let mut config = RemoteAgentConfig::default();
    assert_eq!(config.effective_timeout(), Duration::from_secs(60));
    config.timeout_secs = 0.0;
    assert_eq!(config.effective_timeout(), Duration::from_secs(60));
    config.timeout_secs = -1.0;
    assert_eq!(config.effective_timeout(), Duration::from_secs(60));
    config.timeout_secs = f64::NAN;
    assert_eq!(config.effective_timeout(), Duration::from_secs(60));
    config.timeout_secs = f64::INFINITY;
    assert_eq!(config.effective_timeout(), Duration::from_secs(60));
    config.timeout_secs = 1.5;
    assert_eq!(config.effective_timeout(), Duration::from_millis(1_500));
}

#[test]
fn protocol_wire_names_and_default() {
    assert_eq!(RemoteProtocol::default(), RemoteProtocol::Http);
    for (protocol, name) in [
        (RemoteProtocol::Http, "http"),
        (RemoteProtocol::Mq, "mq"),
        (RemoteProtocol::A2a, "a2a"),
    ] {
        assert_eq!(serde_json::to_value(protocol).unwrap(), json!(name));
        let parsed: RemoteProtocol = serde_json::from_value(json!(name)).unwrap();
        assert_eq!(parsed, protocol);
        assert_eq!(protocol.as_str(), name);
        assert_eq!(protocol.to_string(), name);
    }
}

#[test]
fn urls_follow_the_documented_layout() {
    let config = RemoteAgentConfig::new("assistant", "http://127.0.0.1:8080");
    assert_eq!(
        config.invoke_url().unwrap(),
        "http://127.0.0.1:8080/agents/assistant/invoke"
    );
    assert_eq!(
        config.stream_url().unwrap(),
        "http://127.0.0.1:8080/agents/assistant/stream"
    );
    assert_eq!(
        config.cancel_url().unwrap().unwrap(),
        "http://127.0.0.1:8080/agents/assistant/cancel"
    );
}

#[test]
fn urls_tolerate_extra_and_missing_slashes() {
    let joined = |base: &str, path: &str| {
        RemoteAgentConfig::new("a", base)
            .with_paths(path, path)
            .invoke_url()
            .unwrap()
    };
    assert_eq!(
        joined("http://host:8080/", "/agents/{agent_id}/invoke"),
        "http://host:8080/agents/a/invoke"
    );
    assert_eq!(
        joined("http://host:8080//", "//agents//{agent_id}//invoke"),
        "http://host:8080/agents/a/invoke"
    );
    assert_eq!(
        joined("https://host/api/v1", "agents/{agent_id}/invoke"),
        "https://host/api/v1/agents/a/invoke"
    );
}

#[test]
fn empty_base_url_is_a_validation_error() {
    for config in [
        RemoteAgentConfig::new("a", ""),
        RemoteAgentConfig::new("a", "   "),
    ] {
        for error in [
            config.invoke_url().unwrap_err(),
            config.stream_url().unwrap_err(),
            config.cancel_url().unwrap_err(),
        ] {
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(
                error.to_string().contains("base_url"),
                "unexpected message: {error}"
            );
        }
    }
}

#[test]
fn empty_agent_id_needing_a_placeholder_is_a_validation_error() {
    let config = RemoteAgentConfig::new("", "http://host:8080");
    let error = config.invoke_url().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("agent_id cannot be empty"),
        "unexpected message: {error}"
    );
    assert!(config.stream_url().is_err());
    assert!(config.cancel_url().is_err());
}

#[test]
fn paths_without_a_placeholder_need_no_agent_id() {
    let config = RemoteAgentConfig::new("", "http://host:8080")
        .with_paths("/healthz", "/events")
        .with_cancel_path(Some("/stop".to_owned()));
    assert_eq!(config.invoke_url().unwrap(), "http://host:8080/healthz");
    assert_eq!(config.stream_url().unwrap(), "http://host:8080/events");
    assert_eq!(
        config.cancel_url().unwrap().as_deref(),
        Some("http://host:8080/stop")
    );
}

#[test]
fn unsupported_protocols_name_the_deferred_transport() {
    let error = ensure_supported(RemoteProtocol::Mq).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.to_string();
    assert!(message.contains("remote client protocol 'mq'"), "{message}");
    assert!(message.contains("Pulsar/ZeroMQ"), "{message}");
    assert!(message.contains("RemoteProtocol::Http"), "{message}");

    let error = ensure_supported(RemoteProtocol::A2a).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.to_string();
    assert!(
        message.contains("remote client protocol 'a2a'"),
        "{message}"
    );
    assert!(message.contains("A2A"), "{message}");
    assert!(message.contains("openjiuwen-extensions"), "{message}");

    assert!(ensure_supported(RemoteProtocol::Http).is_ok());
}

#[test]
fn factory_builds_http_and_rejects_deferred_transports() {
    let config = RemoteAgentConfig::new("assistant", "http://127.0.0.1:8080");
    for protocol in [RemoteProtocol::Mq, RemoteProtocol::A2a] {
        let error = create_remote_client(protocol, config.clone())
            .err()
            .expect("a deferred transport cannot build a client");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let error = protocol
            .client(config.clone())
            .err()
            .expect("a deferred transport cannot build a client");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let error = RemoteAgent::with_protocol(config.clone(), protocol).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }
    let client = create_remote_client(RemoteProtocol::Http, config.clone()).unwrap();
    assert!(!client.is_started());
    let agent = RemoteAgent::with_protocol(config, RemoteProtocol::Http).unwrap();
    assert_eq!(agent.protocol(), RemoteProtocol::Http);
}

#[test]
fn status_mapping_splits_not_found_from_transport_failures() {
    let not_found = map_status("a", 404, "http://host/agents/a/invoke", "");
    assert_eq!(not_found.kind(), ErrorKind::NotFound);
    assert!(
        not_found
            .to_string()
            .contains("http://host/agents/a/invoke"),
        "unexpected message: {not_found}"
    );
    let not_found = map_status("a", 404, "http://host/x", "{\"message\":\"no such agent\"}");
    assert!(not_found.to_string().contains("no such agent"));

    for status in [400, 401, 500, 503] {
        let error = map_status("a", status, "http://host/x", "boom");
        assert_eq!(error.kind(), ErrorKind::Transport, "for status {status}");
        let message = error.to_string();
        assert!(message.contains(&status.to_string()), "{message}");
        assert!(message.contains("http://host/x"), "{message}");
        assert!(message.contains("boom"), "{message}");
    }
}

#[test]
fn body_snippets_collapse_whitespace_and_truncate() {
    assert_eq!(body_snippet("  a\n\tb   c  "), "a b c");
    assert_eq!(body_snippet(""), "");
    let long = "é".repeat(BODY_SNIPPET_CHARS * 2);
    let snippet = body_snippet(&long);
    assert_eq!(snippet.chars().count(), BODY_SNIPPET_CHARS);
    assert!(snippet.ends_with('…'));
    let exact = "x".repeat(BODY_SNIPPET_CHARS);
    assert_eq!(body_snippet(&exact), exact);
}

#[test]
fn stream_lines_map_to_items_ends_and_errors() {
    match parse_stream_line(&chunk_line(0, "he")) {
        StreamLine::Chunk(chunk) => {
            assert_eq!(chunk.index, 0);
            assert_eq!(chunk.kind, openjiuwen_agent::output::LLM_OUTPUT);
        }
        other => panic!("expected a chunk, got {other:?}"),
    }
    assert!(matches!(
        parse_stream_line(&frame_line(&SpawnMessage::new(
            SpawnMessageKind::Done,
            Value::Null
        ))),
        StreamLine::End
    ));
    assert!(matches!(
        parse_stream_line(&frame_line(&SpawnMessage::new(
            SpawnMessageKind::Output,
            json!({"output": "done"})
        ))),
        StreamLine::End
    ));
    assert!(matches!(parse_stream_line(""), StreamLine::Skip));
    assert!(matches!(parse_stream_line("   \r\n"), StreamLine::Skip));
    assert!(matches!(
        parse_stream_line("keep-alive: ping"),
        StreamLine::Skip
    ));
    assert!(matches!(
        parse_stream_line(&frame_line(&SpawnMessage::new(
            SpawnMessageKind::HealthCheck,
            Value::Null
        ))),
        StreamLine::Skip
    ));
}

#[test]
fn error_frames_carry_the_remote_message() {
    let message = |payload: Value| match parse_stream_line(&frame_line(&SpawnMessage::new(
        SpawnMessageKind::Error,
        payload,
    ))) {
        StreamLine::Error(error) => error.to_string(),
        other => panic!("expected an error, got {other:?}"),
    };
    assert!(message(json!({"message": "boom"})).contains("boom"));
    assert!(message(json!("plain boom")).contains("plain boom"));
    assert!(message(json!({"code": 7})).contains('7'));
    assert!(message(Value::Null).contains("no error detail"));
}

#[test]
fn valid_json_that_is_not_a_frame_is_a_validation_error() {
    for line in ["42", "{\"not\":\"a frame\"}", "{\"kind\":\"teleport\"}"] {
        match parse_stream_line(line) {
            StreamLine::Error(error) => assert_eq!(error.kind(), ErrorKind::Validation),
            other => panic!("expected a validation error for {line}, got {other:?}"),
        }
    }
}

#[test]
fn a_stream_chunk_without_an_agent_chunk_payload_is_a_validation_error() {
    let line = frame_line(&SpawnMessage::new(
        SpawnMessageKind::StreamChunk,
        json!({"nope": true}),
    ));
    match parse_stream_line(&line) {
        // Any JSON object deserializes into the defaulted `AgentChunk`; a
        // non-object payload is the case that must be rejected.
        StreamLine::Chunk(chunk) => assert_eq!(chunk.index, 0),
        other => panic!("unexpected outcome: {other:?}"),
    }
    let line = frame_line(&SpawnMessage::new(SpawnMessageKind::StreamChunk, json!(5)));
    match parse_stream_line(&line) {
        StreamLine::Error(error) => assert_eq!(error.kind(), ErrorKind::Validation),
        other => panic!("unexpected outcome: {other:?}"),
    }
}

#[test]
fn take_line_splits_on_newlines_and_keeps_the_partial_tail() {
    let mut buffer = b"one\ntwo\r\npartial".to_vec();
    assert_eq!(take_line(&mut buffer).as_deref(), Some("one\n"));
    assert_eq!(take_line(&mut buffer).as_deref(), Some("two\r\n"));
    assert_eq!(take_line(&mut buffer), None);
    assert_eq!(buffer, b"partial");
    buffer.push(b'\n');
    assert_eq!(take_line(&mut buffer).as_deref(), Some("partial\n"));
    assert_eq!(buffer.len(), 0);
}

#[test]
fn remote_ids_come_from_class_module_or_init_kwargs() {
    let spec = SpawnSpec::class_agent("pkg.mod", "HelperAgent");
    assert_eq!(remote_agent_id(&spec).unwrap(), "HelperAgent");

    let spec = SpawnSpec {
        agent_class: Some("  ".to_owned()),
        agent_module: Some("pkg.mod".to_owned()),
        ..SpawnSpec::default()
    };
    assert_eq!(remote_agent_id(&spec).unwrap(), "pkg.mod");

    let spec = SpawnSpec {
        init_kwargs: json!({"agent_id": "remote-1"}),
        ..SpawnSpec::default()
    };
    assert_eq!(remote_agent_id(&spec).unwrap(), "remote-1");

    let spec = SpawnSpec {
        init_kwargs: json!({"agent_id": "  "}),
        ..SpawnSpec::default()
    };
    let error = remote_agent_id(&spec).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error
            .to_string()
            .contains("no agent_class/agent_module/agent_id"),
        "unexpected message: {error}"
    );
}

#[test]
fn spawner_config_for_applies_defaults_headers_and_timeout() {
    let spawner = RemoteSpawner::new("http://127.0.0.1:8080")
        .with_timeout(Duration::from_secs(5))
        .with_header("Authorization", "Bearer token")
        .with_header("X-Team", "red");
    assert_eq!(spawner.base_url(), "http://127.0.0.1:8080");
    let config = spawner.config_for("assistant");
    assert_eq!(config.agent_id, "assistant");
    assert_eq!(config.base_url, "http://127.0.0.1:8080");
    assert_eq!(config.session_id, None);
    assert_eq!(
        config.headers.get("Authorization").map(String::as_str),
        Some("Bearer token")
    );
    assert_eq!(
        config.headers.get("X-Team").map(String::as_str),
        Some("red")
    );
    assert_eq!(config.effective_timeout(), Duration::from_secs(5));
    assert_eq!(
        config.invoke_url().unwrap(),
        "http://127.0.0.1:8080/agents/assistant/invoke"
    );

    let plain = RemoteSpawner::new("http://host").config_for("a");
    assert!((plain.timeout_secs - DEFAULT_REMOTE_TIMEOUT_SECS).abs() < f64::EPSILON);
    assert!(plain.headers.is_empty());
}

#[tokio::test]
async fn spawner_spawns_a_remote_agent_from_the_spec() {
    let spawner = RemoteSpawner::new("http://127.0.0.1:9").with_header("X-Team", "red");
    let spec = SpawnSpec::class_agent("pkg.mod", "HelperAgent").with_session_id("s-1");
    let agent = spawner.spawn(spec).await.unwrap();
    assert_eq!(agent.id(), "HelperAgent");
    assert_eq!(agent.pid(), None);
    assert!(!agent.is_alive());
}

#[tokio::test]
async fn spawner_reports_a_spec_without_an_address() {
    let spawner = RemoteSpawner::new("http://127.0.0.1:9");
    let error = spawner
        .spawn(SpawnSpec::team_agent(json!({"name": "crew"})))
        .await
        .err()
        .expect("a team payload carries no remote agent id");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error
            .to_string()
            .contains("no agent_class/agent_module/agent_id"),
        "unexpected message: {error}"
    );
}

#[tokio::test]
async fn spawner_rejects_an_empty_base_url() {
    let spawner = RemoteSpawner::new("");
    let error = spawner
        .spawn(SpawnSpec::class_agent("pkg.mod", "HelperAgent"))
        .await
        .err()
        .expect("an empty base url cannot address an agent");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn lifecycle_toggles_is_started_and_needs_no_network() {
    let agent = RemoteAgent::new(RemoteAgentConfig::new("a", "http://127.0.0.1:9"));
    assert!(!agent.is_started());
    assert!(agent.is_stopped());
    agent.start().await.unwrap();
    assert!(agent.is_started());
    agent.start().await.unwrap();
    assert!(agent.is_started());
    assert!(agent.is_alive());
    assert_eq!(agent.pid(), None);
    agent.stop().await.unwrap();
    assert!(!agent.is_started());
    agent.stop().await.unwrap();
    assert!(agent.shutdown(Duration::from_secs(1)).await.unwrap());
    assert!(!agent.is_alive());
    agent.start().await.unwrap();
    agent.force_kill().await.unwrap();
    assert!(!agent.is_started());
    assert_eq!(agent.agent_id(), "a");
    assert_eq!(agent.protocol(), RemoteProtocol::Http);
}

#[tokio::test]
async fn invoke_posts_the_session_body_and_parses_the_output() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/agents/assistant/invoke"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"output": "hi there", "result_type": "answer"})),
        )
        .mount(&server)
        .await;
    let agent =
        RemoteAgent::new(RemoteAgentConfig::new("assistant", server.uri()).with_session_id("s-1"));
    let output = agent.invoke(json!({"query": "ping"}), None).await.unwrap();
    assert_eq!(output.output, "hi there");

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["session_id"], json!("s-1"));
    assert_eq!(body["inputs"], json!({"query": "ping"}));

    let output = SpawnedAgent::send(&agent, json!({"query": "ping"}))
        .await
        .unwrap();
    assert_eq!(output.output, "hi there");
}

#[tokio::test]
async fn invoke_sends_a_null_session_when_unset() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/agents/a/invoke"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"output": "ok"})))
        .mount(&server)
        .await;
    let agent = server_agent(&server, "a");
    agent.invoke(json!({}), None).await.unwrap();
    let requests = server.received_requests().await.unwrap();
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["session_id"], Value::Null);
}

#[tokio::test]
async fn invoke_maps_http_failures_onto_error_kinds() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/agents/missing/invoke"))
        .respond_with(ResponseTemplate::new(404).set_body_string("no such agent"))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/agents/boom/invoke"))
        .respond_with(ResponseTemplate::new(500).set_body_string("kaboom"))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/agents/garbage/invoke"))
        .respond_with(ResponseTemplate::new(200).set_body_string("<html>nope</html>"))
        .mount(&server)
        .await;

    let error = server_agent(&server, "missing")
        .invoke(json!({}), None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.to_string().contains("no such agent"), "{error}");
    assert!(
        error.to_string().contains("/agents/missing/invoke"),
        "{error}"
    );

    let error = server_agent(&server, "boom")
        .invoke(json!({}), None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(error.to_string().contains("500"), "{error}");
    assert!(error.to_string().contains("kaboom"), "{error}");

    let error = server_agent(&server, "garbage")
        .invoke(json!({}), None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("/agents/garbage/invoke"),
        "{error}"
    );
    assert!(error.to_string().contains("<html>nope</html>"), "{error}");
}

#[tokio::test]
async fn invoke_reports_an_unreachable_agent_as_a_transport_error() {
    // Port 1 on loopback is closed: the request fails without any DNS.
    let agent = RemoteAgent::new(RemoteAgentConfig::new("a", "http://127.0.0.1:1"));
    let error = agent.invoke(json!({}), None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(error.to_string().contains("remote agent invoke"), "{error}");
}

#[tokio::test]
async fn stream_yields_chunks_and_ends_on_done() {
    let server = MockServer::start().await;
    let body = [
        chunk_line(0, "he"),
        chunk_line(1, "llo"),
        "keep-alive".to_owned(),
        frame_line(&SpawnMessage::new(SpawnMessageKind::Done, Value::Null)),
    ]
    .join("\n");
    Mock::given(method("POST"))
        .and(path("/agents/a/stream"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(&server)
        .await;

    let agent = server_agent(&server, "a");
    let items: Vec<_> = RemoteClient::stream(&agent, json!({"query": "hi"}), None)
        .collect()
        .await;
    assert_eq!(items.len(), 2);
    let texts: Vec<Value> = items
        .into_iter()
        .map(|item| item.unwrap().payload)
        .collect();
    assert_eq!(texts[0], json!({"content": "he"}));
    assert_eq!(texts[1], json!({"content": "llo"}));
}

#[tokio::test]
async fn stream_parses_a_trailing_line_without_a_newline() {
    let server = MockServer::start().await;
    let body = chunk_line(0, "only");
    Mock::given(method("POST"))
        .and(path("/agents/a/stream"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(&server)
        .await;
    let agent = server_agent(&server, "a");
    let items: Vec<_> = RemoteClient::stream(&agent, json!({}), None)
        .collect()
        .await;
    assert_eq!(items.len(), 1);
    assert_eq!(
        items.into_iter().next().unwrap().unwrap().payload,
        json!({"content": "only"})
    );
}

#[tokio::test]
async fn stream_maps_an_error_frame_to_one_error_item() {
    let server = MockServer::start().await;
    let body = [
        chunk_line(0, "he"),
        frame_line(&SpawnMessage::new(
            SpawnMessageKind::Error,
            json!({"message": "boom"}),
        )),
        chunk_line(1, "never"),
    ]
    .join("\n");
    Mock::given(method("POST"))
        .and(path("/agents/a/stream"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(&server)
        .await;
    let agent = server_agent(&server, "a");
    let items: Vec<_> = RemoteClient::stream(&agent, json!({}), None)
        .collect()
        .await;
    assert_eq!(items.len(), 2);
    assert!(items[0].is_ok());
    let error = items[1].as_ref().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(error.to_string().contains("boom"), "{error}");
}

#[tokio::test]
async fn stream_maps_a_bad_status_to_one_error_item() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/agents/a/stream"))
        .respond_with(ResponseTemplate::new(503).set_body_string("overloaded"))
        .mount(&server)
        .await;
    let agent = server_agent(&server, "a");
    let items: Vec<_> = RemoteClient::stream(&agent, json!({}), None)
        .collect()
        .await;
    assert_eq!(items.len(), 1);
    let error = items[0].as_ref().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(error.to_string().contains("503"), "{error}");
    assert!(error.to_string().contains("overloaded"), "{error}");
}

#[tokio::test]
async fn stream_rejects_an_unusable_url_as_one_error_item() {
    let agent = RemoteAgent::new(RemoteAgentConfig::new("", "http://127.0.0.1:9"));
    let items: Vec<_> = RemoteClient::stream(&agent, json!({}), None)
        .collect()
        .await;
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].as_ref().unwrap_err().kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn cancel_task_posts_the_task_id() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/agents/a/cancel"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
        .mount(&server)
        .await;
    let agent = server_agent(&server, "a");
    agent.cancel_task("task-1").await.unwrap();
    let requests = server.received_requests().await.unwrap();
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body, json!({"task_id": "task-1"}));
}

#[tokio::test]
async fn cancel_task_without_a_cancel_path_is_unsupported() {
    let agent =
        RemoteAgent::new(RemoteAgentConfig::new("a", "http://127.0.0.1:9").with_cancel_path(None));
    let error = agent.cancel_task("task-1").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        error.to_string().contains("remote agent cancel_task"),
        "{error}"
    );
    assert!(error.to_string().contains("cancel_path"), "{error}");
}

#[tokio::test]
async fn cancel_task_maps_a_missing_task_to_not_found() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/agents/a/cancel"))
        .respond_with(ResponseTemplate::new(404).set_body_string("unknown task"))
        .mount(&server)
        .await;
    let agent = server_agent(&server, "a");
    let error = agent.cancel_task("task-1").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.to_string().contains("unknown task"), "{error}");
}

#[tokio::test]
async fn custom_paths_are_honoured() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v2/run/assistant"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"output": "ok"})))
        .mount(&server)
        .await;
    let agent = RemoteAgent::new(
        RemoteAgentConfig::new("assistant", server.uri())
            .with_paths("v2/run/{agent_id}", "v2/events/{agent_id}"),
    );
    let output = agent.invoke(json!({}), None).await.unwrap();
    assert_eq!(output.output, "ok");
}
