//! End-to-end suites for the `a2a` module (public API only).
//!
//! These two submodules mirror the crate-level integration targets
//! `tests/a2a_client.rs` and `tests/a2a_server.rs`: to promote one, copy the
//! module body into that file, add `#![cfg(feature = "a2a")]` at the top, and
//! rewrite `crate::a2a::` paths to `openjiuwen_extensions::a2a::`.
#![cfg(feature = "a2a")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

/// Client-side end-to-end coverage.
///
/// [`client::A2aClient`] is driven against a `wiremock` peer and asserted from
/// both ends: the folded [`transform::AgentResult`] the caller sees, and the
/// JSON-RPC envelope the peer actually received. [`client::A2aRemoteClient`] is
/// then driven through the endpoint its agent card advertises.
mod client_suite {
    use crate::a2a::client::{A2aClient, A2aClientConfig, A2aRemoteClient, RemoteClientConfig};
    use crate::a2a::transform::{AgentResult, AgentStatus};
    use crate::a2a::types::{AgentCard, AgentInterface};
    use futures::StreamExt;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::{json, Value};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// The JSON-RPC endpoint of `server`.
    fn rpc_url(server: &MockServer) -> String {
        format!("{}/a2a/jsonrpc/", server.uri())
    }

    /// A client posting to the JSON-RPC endpoint of `server`.
    fn client(server: &MockServer) -> A2aClient {
        A2aClient::new(A2aClientConfig::new(rpc_url(server))).expect("the client builds")
    }

    /// Serializes `frames` as one SSE `data:` frame each.
    fn sse_body(frames: &[Value]) -> String {
        let mut body = String::new();
        for frame in frames {
            body.push_str("data: ");
            body.push_str(&frame.to_string());
            body.push_str("\n\n");
        }
        body
    }

    /// The single request `server` received, decoded as JSON.
    async fn received_body(server: &MockServer) -> Value {
        let requests = server
            .received_requests()
            .await
            .expect("the mock records requests");
        assert_eq!(requests.len(), 1, "exactly one request was sent");
        serde_json::from_slice(&requests[0].body).expect("the request body is JSON")
    }

    #[tokio::test]
    async fn send_returns_a_task_and_folds_to_an_agent_result() {
        // ports `A2AClient.invoke` (`_merge_agent_results` + `_with_session_id`)
        // over a `Task` response.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/a2a/jsonrpc/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "id": "task-1",
                    "contextId": "ctx-1",
                    "status": { "state": "TASK_STATE_COMPLETED" },
                    "artifacts": [{ "artifactId": "a1", "parts": [{ "text": "done" }] }],
                    "history": [],
                    "metadata": { "k": "v" }
                }
            })))
            .mount(&server)
            .await;

        let result = client(&server)
            .invoke(json!({ "query": "hi", "conversation_id": "sess-1" }))
            .await
            .expect("the send succeeds");

        assert_eq!(result.status, AgentStatus::Completed);
        assert_eq!(result.task_id.as_deref(), Some("task-1"));
        // The request session id wins over the task's own `contextId`.
        assert_eq!(result.session_id.as_deref(), Some("sess-1"));
        assert_eq!(result.artifacts.len(), 1);
        assert_eq!(result.artifacts[0].artifact_id.as_deref(), Some("a1"));
        assert_eq!(result.artifacts[0].parts.len(), 1);
        assert_eq!(result.artifacts[0].parts[0].text.as_deref(), Some("done"));
        assert_eq!(result.metadata.get("k"), Some(&json!("v")));

        let body = received_body(&server).await;
        assert_eq!(body["jsonrpc"], "2.0");
        assert_eq!(body["method"], "SendMessage");
        assert_eq!(body["params"]["message"]["role"], "ROLE_USER");
        assert_eq!(body["params"]["message"]["parts"][0]["text"], "hi");
        assert_eq!(body["params"]["message"]["contextId"], "sess-1");
        assert!(
            body["params"]["message"]["messageId"]
                .as_str()
                .is_some_and(|id| !id.is_empty()),
            "the client generates a message id: {body}"
        );
    }

    #[tokio::test]
    async fn stream_yields_one_result_per_frame() {
        // ports `A2AClient.stream` yielding one `from_a2a_response` per SSE event.
        let server = MockServer::start().await;
        let body = sse_body(&[
            json!({
                "jsonrpc": "2.0", "id": 1,
                "result": {
                    "taskId": "task-1", "contextId": "ctx-1",
                    "status": { "state": "TASK_STATE_WORKING" }
                }
            }),
            json!({
                "jsonrpc": "2.0", "id": 2,
                "result": {
                    "taskId": "task-1", "contextId": "ctx-1", "lastChunk": false,
                    "artifact": { "artifactId": "a1", "parts": [{ "text": "chunk" }] }
                }
            }),
            json!({
                "jsonrpc": "2.0", "id": 3,
                "result": {
                    "taskId": "task-1", "contextId": "ctx-1",
                    "status": { "state": "TASK_STATE_COMPLETED" }
                }
            }),
        ]);
        Mock::given(method("POST"))
            .and(path("/a2a/jsonrpc/"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;

        let frames: Vec<AgentResult> = client(&server)
            .stream(json!({ "query": "go", "sessionId": "s-1" }))
            .map(|frame| frame.expect("every frame decodes"))
            .collect()
            .await;

        assert_eq!(frames.len(), 3, "one result per SSE frame");
        assert_eq!(frames[0].status, AgentStatus::Working);
        assert_eq!(frames[0].artifacts, Vec::new());
        assert_eq!(frames[1].status, AgentStatus::Working);
        assert_eq!(frames[1].artifacts.len(), 1);
        assert_eq!(frames[1].artifacts[0].artifact_id.as_deref(), Some("a1"));
        assert_eq!(
            frames[1].artifacts[0].parts[0].text.as_deref(),
            Some("chunk")
        );
        assert_eq!(frames[2].status, AgentStatus::Completed);
        for frame in &frames {
            assert_eq!(frame.session_id.as_deref(), Some("s-1"));
            assert_eq!(frame.task_id.as_deref(), Some("task-1"));
        }

        let body = received_body(&server).await;
        assert_eq!(body["method"], "SendStreamingMessage");
    }

    #[tokio::test]
    async fn cancel_task_sends_the_pascal_case_method_and_maps_canceled() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/a2a/jsonrpc/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "id": "task-9",
                    "contextId": "ctx-9",
                    "status": { "state": "TASK_STATE_CANCELED" }
                }
            })))
            .mount(&server)
            .await;

        let result = client(&server)
            .cancel_task("task-9", Some("tenant-7"))
            .await
            .expect("the cancel succeeds");

        assert_eq!(result.status, AgentStatus::Canceled);
        assert_eq!(result.task_id.as_deref(), Some("task-9"));

        let body = received_body(&server).await;
        assert_eq!(body["method"], "CancelTask");
        assert_eq!(body["params"]["id"], "task-9");
        assert_eq!(body["params"]["tenant"], "tenant-7");
    }

    #[tokio::test]
    async fn json_rpc_error_bodies_map_to_typed_errors() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/a2a/jsonrpc/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0",
                "id": 1,
                "error": { "code": -32001, "message": "Task not found" }
            })))
            .mount(&server)
            .await;

        let error = client(&server)
            .invoke(json!({ "query": "hi" }))
            .await
            .expect_err("the JSON-RPC error is reported");

        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(
            error.to_string().contains("Task not found"),
            "the peer message survives: {error}"
        );
    }

    #[tokio::test]
    async fn a_non_2xx_response_is_a_transport_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/a2a/jsonrpc/"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let error = client(&server)
            .invoke(json!({ "query": "hi" }))
            .await
            .expect_err("a 500 is not a JSON-RPC answer");

        assert_eq!(error.kind(), ErrorKind::Transport);
    }

    #[tokio::test]
    async fn remote_client_requires_a_card_and_the_a2a_protocol() {
        // A card is mandatory, and only the A2A protocol is spoken.
        let error = A2aRemoteClient::new(RemoteClientConfig {
            protocol: String::from("A2A"),
            url: String::from("http://127.0.0.1:9"),
            ..RemoteClientConfig::default()
        })
        .expect_err("the card is required");
        assert_eq!(error.kind(), ErrorKind::Validation);

        let error = A2aRemoteClient::new(RemoteClientConfig {
            protocol: String::from("GRPC"),
            card: Some(AgentCard::default()),
            ..RemoteClientConfig::default()
        })
        .expect_err("gRPC is rejected");
        assert_eq!(error.kind(), ErrorKind::Validation);

        let client = A2aRemoteClient::new(RemoteClientConfig {
            protocol: String::from("A2A"),
            card: Some(AgentCard::default()),
            url: String::from("http://127.0.0.1:9"),
            ..RemoteClientConfig::default()
        })
        .expect("a card plus a base url is enough");
        assert_eq!(client.rpc_url(), "http://127.0.0.1:9/a2a/jsonrpc/");
        assert!(!client.is_started());
        client.start().await.expect("starting is infallible");
        assert!(client.is_started());
        client.stop().await;
        assert!(!client.is_started());
    }

    #[tokio::test]
    async fn remote_client_invokes_through_the_card_interface() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/a2a/jsonrpc/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "id": "task-remote",
                    "contextId": "ctx-remote",
                    "status": { "state": "TASK_STATE_COMPLETED" },
                    "artifacts": [{ "artifactId": "a1", "parts": [{ "text": "pong" }] }]
                }
            })))
            .mount(&server)
            .await;

        let card = AgentCard {
            supported_interfaces: vec![AgentInterface {
                url: rpc_url(&server),
                protocol_binding: String::from("JSONRPC"),
                protocol_version: String::from("1.0"),
                tenant: None,
            }],
            ..AgentCard::default()
        };
        let client = A2aRemoteClient::new(RemoteClientConfig {
            protocol: String::from("A2A"),
            card: Some(card),
            url: String::new(),
            ..RemoteClientConfig::default()
        })
        .expect("the card publishes the endpoint");
        assert_eq!(client.rpc_url(), rpc_url(&server));

        let result = client
            .invoke(json!({ "query": "ping", "sessionId": "s-2" }), None)
            .await
            .expect("the remote call succeeds");

        assert_eq!(result.status, AgentStatus::Completed);
        assert_eq!(result.session_id.as_deref(), Some("s-2"));
        assert_eq!(result.task_id.as_deref(), Some("task-remote"));
        assert_eq!(result.artifacts[0].parts[0].text.as_deref(), Some("pong"));
    }
}

/// Server-side end-to-end coverage.
///
/// A real [`server::A2aServer`] is driven both through its axum router (with
/// `tower`'s in-process `oneshot`) and over a real loopback socket, asserting
/// the client-visible JSON-RPC envelopes and SSE frames.
mod server_suite {
    use std::sync::Arc;

    use async_trait::async_trait;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use openjiuwen_common::error::{Error, Result};
    use serde_json::{json, Value};
    use tokio_util::sync::CancellationToken;
    use tower::ServiceExt;

    use crate::a2a::jsonrpc::JsonRpcResponse;
    use crate::a2a::server::{
        A2aExecutionContext, A2aExecutor, A2aServer, A2aServerConfig, EventSink,
    };
    use crate::a2a::types::{
        AgentCapabilities, AgentCard, Artifact, Message, Part, Role, Task, TaskState, TaskStatus,
    };

    /// The card every server in this suite publishes.
    fn card() -> AgentCard {
        AgentCard {
            name: String::from("test-agent"),
            description: String::from("d"),
            capabilities: AgentCapabilities {
                streaming: true,
                push_notifications: false,
            },
            default_input_modes: vec![String::from("text/plain")],
            default_output_modes: vec![String::from("text/plain")],
            ..AgentCard::default()
        }
    }

    /// An executor that publishes one artifact in two chunks, or fails.
    #[derive(Default)]
    struct FakeExecutor {
        /// When set, `execute` returns [`Error::Tool`] instead of artifacts.
        fail: bool,
    }

    #[async_trait]
    impl A2aExecutor for FakeExecutor {
        async fn execute(&self, _ctx: A2aExecutionContext, sink: &EventSink) -> Result<()> {
            if self.fail {
                return Err(Error::Tool(String::from("boom")));
            }
            sink.artifact(
                Artifact {
                    artifact_id: String::from("a1"),
                    parts: vec![Part::text("hello")],
                    ..Artifact::default()
                },
                false,
            );
            sink.artifact(
                Artifact {
                    artifact_id: String::from("a1"),
                    parts: vec![Part::text("hello again")],
                    ..Artifact::default()
                },
                true,
            );
            Ok(())
        }
    }

    /// A server over `card()` and `executor`.
    fn server_with(executor: FakeExecutor) -> A2aServer {
        A2aServer::new(A2aServerConfig::new(card()), Arc::new(executor))
            .expect("the card advertises a JSON-RPC transport")
    }

    /// The `SendMessage` params carrying `text` as a user message.
    fn send_params(text: &str) -> Value {
        json!({
            "message": {
                "messageId": "m1",
                "role": "ROLE_USER",
                "parts": [{ "text": text }]
            }
        })
    }

    /// POSTs `body` to the JSON-RPC route and returns the raw response.
    async fn post_response(server: &A2aServer, body: &Value) -> axum::response::Response {
        server
            .router()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/a2a/jsonrpc/")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .expect("the request builds"),
            )
            .await
            .expect("the router answers")
    }

    /// POSTs `body` to the JSON-RPC route and decodes the JSON-RPC envelope.
    async fn post(server: &A2aServer, body: &Value) -> JsonRpcResponse {
        let response = post_response(server, body).await;
        assert_eq!(response.status(), StatusCode::OK, "JSON-RPC answers 200");
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("the body is readable")
            .to_bytes();
        serde_json::from_slice(&bytes).expect("the body is a JSON-RPC response")
    }

    /// GETs `uri` from the router and returns the status and decoded body.
    async fn get(server: &A2aServer, uri: &str) -> (StatusCode, Value) {
        let response = server
            .router()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(uri)
                    .body(Body::empty())
                    .expect("the request builds"),
            )
            .await
            .expect("the router answers");
        let status = response.status();
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("the body is readable")
            .to_bytes();
        let body = serde_json::from_slice(&bytes).expect("the body is JSON");
        (status, body)
    }

    /// Decodes every SSE frame of `response` as a JSON-RPC envelope.
    async fn sse_frames(response: axum::response::Response) -> Vec<JsonRpcResponse> {
        assert_eq!(
            response
                .headers()
                .get("content-type")
                .and_then(|value| value.to_str().ok()),
            Some("text/event-stream"),
            "a streaming call answers with an event stream"
        );
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("the body is readable")
            .to_bytes();
        let text = String::from_utf8(bytes.to_vec()).expect("the SSE body is UTF-8");
        text.split("\n\n")
            .filter(|frame| !frame.is_empty())
            .map(|frame| {
                let data = frame
                    .strip_prefix("data: ")
                    .expect("every frame carries a data line");
                serde_json::from_str::<JsonRpcResponse>(data)
                    .expect("every frame is a JSON-RPC response")
            })
            .collect()
    }

    /// The `status.state` of a successful frame.
    fn state_of(frame: &JsonRpcResponse) -> &str {
        frame.result.as_ref().expect("the frame carries a result")["status"]["state"]
            .as_str()
            .expect("the frame carries a status state")
    }

    #[tokio::test]
    async fn card_route_serves_the_agent_card() {
        let server = server_with(FakeExecutor::default());

        let (status, card) = get(&server, "/.well-known/agent-card.json").await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(card["name"], "test-agent");
        assert_eq!(card["capabilities"]["streaming"], true);
        assert!(
            card["defaultInputModes"]
                .as_array()
                .expect("the modes are a list")
                .contains(&json!("text/plain")),
            "the default input modes are published: {card}"
        );
    }

    #[tokio::test]
    async fn send_message_and_the_v0_3_alias_both_dispatch() {
        let server = server_with(FakeExecutor::default());

        for method in ["message/send", "SendMessage"] {
            let response = post(
                &server,
                &json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": method,
                    "params": send_params("hi")
                }),
            )
            .await;

            assert!(
                response.error.is_none(),
                "`{method}` succeeded: {response:?}"
            );
            let result = response.result.expect("the call returns a result");
            assert_eq!(
                result["status"]["state"], "TASK_STATE_COMPLETED",
                "`{method}` completed the task"
            );
            let artifact_ids: Vec<&Value> = result["artifacts"]
                .as_array()
                .expect("the task carries artifacts")
                .iter()
                .map(|artifact| &artifact["artifactId"])
                .collect();
            assert_eq!(artifact_ids, vec![&json!("a1")], "`{method}`");
        }
    }

    #[tokio::test]
    async fn get_task_for_an_unknown_id_reports_task_not_found() {
        let server = server_with(FakeExecutor::default());

        let response = post(
            &server,
            &json!({ "jsonrpc": "2.0", "id": 1, "method": "tasks/get", "params": { "id": "nope" } }),
        )
        .await;

        assert!(response.result.is_none());
        let error = response.error.expect("an unknown task is an error");
        assert_eq!(error.code, crate::a2a::jsonrpc::error_code::TASK_NOT_FOUND);
        assert_eq!(error.code, -32001);
    }

    #[tokio::test]
    async fn cancel_task_returns_canceled() {
        let server = server_with(FakeExecutor::default());
        // A synchronous `SendMessage` always publishes a terminal status, so the
        // only task a cancel can legally act on is seeded through the public
        // `TaskStore` SPI: one still `WORKING`, as a live streaming turn has it.
        server.store().put(Task {
            id: String::from("task-cancel"),
            context_id: String::from("ctx-cancel"),
            status: TaskStatus {
                state: TaskState::Working,
                ..TaskStatus::default()
            },
            history: vec![Message {
                message_id: String::from("m-cancel"),
                role: Role::User,
                parts: vec![Part::text("hi")],
                ..Message::default()
            }],
            ..Task::default()
        });

        let response = post(
            &server,
            &json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "CancelTask",
                "params": { "id": "task-cancel" }
            }),
        )
        .await;

        assert!(
            response.error.is_none(),
            "the cancel succeeds: {response:?}"
        );
        let result = response.result.expect("the cancel returns the task");
        assert_eq!(result["status"]["state"], "TASK_STATE_CANCELED");
        assert_eq!(result["id"], "task-cancel");
        assert_eq!(
            server
                .store()
                .get("task-cancel")
                .expect("the cancelled task is stored")
                .status
                .state,
            TaskState::Canceled
        );

        // Cancelling the now-terminal task again is refused.
        let again = post(
            &server,
            &json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "CancelTask",
                "params": { "id": "task-cancel" }
            }),
        )
        .await;
        let error = again.error.expect("a terminal task is not cancelable");
        assert_eq!(error.code, -32002);
    }

    #[tokio::test]
    async fn message_stream_emits_the_python_frame_order() {
        let server = server_with(FakeExecutor::default());

        let response = post_response(
            &server,
            &json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "SendStreamingMessage",
                "params": send_params("hi")
            }),
        )
        .await;
        let frames = sse_frames(response).await;

        assert_eq!(
            frames.len(),
            5,
            "submitted, working, two artifact chunks, completed: {frames:?}"
        );

        assert_eq!(state_of(&frames[0]), "TASK_STATE_SUBMITTED");
        let snapshot = frames[0].result.as_ref().expect("the snapshot is a result");
        assert_eq!(snapshot["history"][0]["role"], "ROLE_USER");
        assert_eq!(snapshot["history"][0]["parts"][0]["text"], "hi");
        assert_eq!(snapshot["artifacts"], json!([]));

        assert_eq!(state_of(&frames[1]), "TASK_STATE_WORKING");

        let first = frames[2]
            .result
            .as_ref()
            .expect("the artifact chunk is a result");
        assert_eq!(first["artifact"]["artifactId"], "a1");
        assert_eq!(first["artifact"]["parts"][0]["text"], "hello");
        assert_eq!(first["lastChunk"], false);

        let second = frames[3]
            .result
            .as_ref()
            .expect("the last chunk is a result");
        assert_eq!(second["artifact"]["artifactId"], "a1");
        assert_eq!(second["artifact"]["parts"][0]["text"], "hello again");
        assert_eq!(second["lastChunk"], true);

        assert_eq!(state_of(&frames[4]), "TASK_STATE_COMPLETED");
        for frame in &frames {
            assert_eq!(frame.id, json!(1), "every frame answers the request id");
        }
    }

    #[tokio::test]
    async fn executor_error_emits_failed_and_returns_an_internal_error() {
        let server = server_with(FakeExecutor { fail: true });

        let response = post(
            &server,
            &json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "SendMessage",
                "params": send_params("hi")
            }),
        )
        .await;

        assert!(response.result.is_none(), "a failure has no result");
        let error = response.error.expect("the executor failure is reported");
        assert_eq!(error.code, crate::a2a::jsonrpc::error_code::INTERNAL);
        assert_eq!(error.code, -32603);
        assert!(
            error.message.contains("boom"),
            "the executor message survives: {}",
            error.message
        );

        let tasks = server.store().all();
        assert_eq!(tasks.len(), 1, "the failed task is stored");
        let task = server
            .store()
            .get(&tasks[0].id)
            .expect("the failed task is readable by id");
        assert_eq!(task.status.state, TaskState::Failed);
    }

    #[tokio::test]
    async fn server_answers_over_a_real_socket() {
        let server = A2aServer::new(
            A2aServerConfig::new(card()).with_bind("127.0.0.1", 0),
            Arc::new(FakeExecutor::default()),
        )
        .expect("the card advertises a JSON-RPC transport");
        let rpc_path = server.config().rpc_path.clone();
        let handle = server
            .serve_in_background(CancellationToken::new())
            .await
            .expect("the server binds an ephemeral port");
        let base = format!("http://{}", handle.local_addr());

        let card_response = reqwest::get(format!("{base}/.well-known/agent-card.json"))
            .await
            .expect("the card route is reachable");
        assert_eq!(card_response.status(), StatusCode::OK);
        let card: AgentCard = card_response.json().await.expect("the card is JSON");
        assert_eq!(card.name, "test-agent");
        assert!(card.capabilities.streaming);

        let response = reqwest::Client::new()
            .post(format!("{base}{rpc_path}"))
            .json(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "SendMessage",
                "params": send_params("hi")
            }))
            .send()
            .await
            .expect("the JSON-RPC route is reachable");
        assert_eq!(response.status(), StatusCode::OK);
        let envelope: JsonRpcResponse = response.json().await.expect("the answer is JSON-RPC");
        assert!(envelope.error.is_none(), "the call succeeds: {envelope:?}");
        let result = envelope.result.expect("the call returns a result");
        assert_eq!(result["status"]["state"], "TASK_STATE_COMPLETED");
        assert_eq!(result["artifacts"][0]["artifactId"], "a1");

        handle.shutdown();
        handle.join().await.expect("the server stops cleanly");
    }
}
