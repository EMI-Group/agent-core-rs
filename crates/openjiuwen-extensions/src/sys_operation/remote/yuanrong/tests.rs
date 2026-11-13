//! Unit tests for the `yuanrong` ADX backend.
//!
//! Python counterparts: the `YuanRong` suites under
//! `tests/unit_tests/extensions/sys_operation/sandbox/`. Every test talks to a
//! loopback `wiremock` server and never reaches a real ADX deployment.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_core::sys_operation::code::types::Language;
use openjiuwen_core::sys_operation::sandbox::config::SandboxLauncherConfig;
use serde_json::{json, Value};
use wiremock::matchers::{method, path, path_regex};
use wiremock::{Mock, MockServer, Request as WireRequest, Respond, ResponseTemplate};

use super::super::transport::base64_encode;
use super::*;

/// The ADX management-plane environment path.
const CREATE_PATH: &str = "/api/environment/v1/environments";

/// Names one card endpoint per test so the process-global environment cache
/// cannot leak an entry from another test whose loopback server reused a port.
static NEXT_ISOLATION: AtomicUsize = AtomicUsize::new(0);

/// A card endpoint pointing at `base_url`.
fn endpoint(base_url: &str) -> SandboxEndpoint {
    let sequence = NEXT_ISOLATION.fetch_add(1, Ordering::SeqCst);
    SandboxEndpoint {
        base_url: base_url.to_string(),
        sandbox_id: None,
        isolation_key: Some(format!("session_pre_deploy_yuanrong_{sequence}")),
    }
}

/// A gateway configuration carrying `extra` as the launcher parameters.
fn gateway_config(extra: &[(&str, Value)]) -> SandboxGatewayConfig {
    let mut params = BTreeMap::new();
    for (key, value) in extra {
        params.insert((*key).to_string(), value.clone());
    }
    SandboxGatewayConfig {
        launcher_config: Some(SandboxLauncherConfig {
            extra_params: params,
            ..SandboxLauncherConfig::default()
        }),
        ..SandboxGatewayConfig::default()
    }
}

/// A backend bound to `server`.
fn backend_for(server: &MockServer, extra: &[(&str, Value)]) -> YuanRongBackend {
    YuanRongBackend::new(&endpoint(&server.uri()), &gateway_config(extra)).expect("backend")
}

/// Mounts a create endpoint that always returns `environment_id`.
async fn mount_create(server: &MockServer, environment_id: &'static str) {
    Mock::given(method("POST"))
        .and(path(CREATE_PATH))
        .respond_with(move |_request: &WireRequest| {
            ResponseTemplate::new(200)
                .set_body_json(json!({ "data": { "environmentId": environment_id } }))
        })
        .mount(server)
        .await;
}

/// Mounts every `/direct/{id}/invoke` request with one responder.
async fn mount_invoke(server: &MockServer, responder: impl Respond + 'static) {
    Mock::given(method("POST"))
        .and(path_regex(r"^/direct/.+/invoke$"))
        .respond_with(responder)
        .mount(server)
        .await;
}

/// The parsed JSON body of one recorded request.
fn request_body(request: &WireRequest) -> Value {
    request.body_json().unwrap_or(Value::Null)
}

/// Every recorded request whose body carries `action`.
async fn action_bodies(server: &MockServer, action: &str) -> Vec<Value> {
    server
        .received_requests()
        .await
        .expect("requests")
        .iter()
        .map(request_body)
        .filter(|body| body["action"] == json!(action))
        .collect()
}

#[test]
fn create_body_defaults_and_launcher_overrides() {
    let backend = YuanRongBackend::new(&endpoint("http://127.0.0.1:9"), &gateway_config(&[]))
        .expect("backend");
    let body = backend.create_body();
    assert_eq!(body["cpu"], json!(1000));
    assert_eq!(body["memory"], json!(1024));
    assert_eq!(body["runtime"], json!("supervisor"));
    assert_eq!(body["idleTimeoutSeconds"], json!(0));
    assert!(body.get("extra_config").is_none());
    let rootfs: Value =
        serde_json::from_str(body["rootfs"].as_str().expect("rootfs string")).expect("json");
    assert_eq!(
        rootfs,
        json!({"type": "image", "imageurl": null, "workdir": null, "mounts": null})
    );

    let backend = YuanRongBackend::new(
        &endpoint("http://127.0.0.1:9"),
        &gateway_config(&[
            ("executor", json!("'Docker'")),
            ("cpu", json!(2000)),
            ("memory", json!(512)),
            ("image", json!("img:1")),
            ("workdir", json!("/work")),
            ("mounts", json!([{"source": "/a", "target": "/b"}])),
        ]),
    )
    .expect("backend");
    let body = backend.create_body();
    assert_eq!(body["runtime"], json!("docker"));
    assert_eq!(body["cpu"], json!(2000));
    assert_eq!(body["memory"], json!(512));
    let rootfs: Value =
        serde_json::from_str(body["rootfs"].as_str().expect("rootfs string")).expect("json");
    assert_eq!(
        rootfs,
        json!({
            "type": "image",
            "imageurl": "img:1",
            "workdir": "/work",
            "mounts": [{"source": "/a", "target": "/b"}],
        })
    );

    let backend = YuanRongBackend::new(
        &endpoint("http://127.0.0.1:9"),
        &gateway_config(&[("rootfs", json!({"image": "img:2", "workdir": "/w"}))]),
    )
    .expect("backend");
    let rootfs: Value = serde_json::from_str(
        backend.create_body()["rootfs"]
            .as_str()
            .expect("rootfs string"),
    )
    .expect("json");
    assert_eq!(rootfs, json!({"imageurl": "img:2", "workdir": "/w"}));

    let backend = YuanRongBackend::new(
        &endpoint("http://127.0.0.1:9"),
        &gateway_config(&[("rootfs", json!("{\"type\": \"image\"}"))]),
    )
    .expect("backend");
    assert_eq!(
        backend.create_body()["rootfs"],
        json!("{\"type\": \"image\"}")
    );

    let backend = YuanRongBackend::new(
        &endpoint("http://127.0.0.1:9"),
        &gateway_config(&[("idle_timeout", json!(-5)), ("user", json!(" alice "))]),
    )
    .expect("backend");
    let body = backend.create_body();
    assert_eq!(body["idleTimeoutSeconds"], json!(0));
    assert_eq!(body["extra_config"], json!({"user": "alice"}));

    let backend = YuanRongBackend::new(
        &endpoint("http://127.0.0.1:9"),
        &gateway_config(&[("executor", json!("bogus"))]),
    )
    .expect("backend");
    assert_eq!(backend.create_body()["runtime"], json!("supervisor"));
}

#[test]
fn a_boolean_idle_timeout_is_a_configuration_error() {
    let error = YuanRongBackend::new(
        &endpoint("http://127.0.0.1:9"),
        &gateway_config(&[("idleTimeoutSeconds", json!(true))]),
    )
    .err()
    .expect("error");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(error.message(), "idleTimeoutSeconds must be an integer");
}

#[tokio::test]
async fn create_posts_the_adx_environment_body() {
    let server = MockServer::start().await;
    mount_create(&server, "env-1").await;
    mount_invoke(
        &server,
        ResponseTemplate::new(200).set_body_json(json!({"data": "6869"})),
    )
    .await;
    let backend = backend_for(&server, &[("cpu", json!(2000)), ("user", json!(" alice "))]);
    assert_eq!(backend.read_bytes("/tmp/a.txt").await.expect("read"), b"hi");

    let requests = server.received_requests().await.expect("requests");
    let create = requests
        .iter()
        .find(|request| request.url.path() == CREATE_PATH)
        .expect("create request");
    assert_eq!(create.method.as_str(), "POST");
    assert_eq!(
        create
            .headers
            .get("accept")
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let body = request_body(create);
    assert_eq!(body["cpu"], json!(2000));
    assert_eq!(body["memory"], json!(1024));
    assert_eq!(body["runtime"], json!("supervisor"));
    assert_eq!(body["idleTimeoutSeconds"], json!(0));
    assert_eq!(body["extra_config"], json!({"user": "alice"}));
    assert!(body["rootfs"].is_string());
}

#[tokio::test]
async fn create_accepts_a_base64_envelope_and_requires_an_environment_id() {
    let server = MockServer::start().await;
    let encoded = base64_encode(json!({"environmentId": "env-b"}).to_string().as_bytes());
    Mock::given(method("POST"))
        .and(path(CREATE_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": encoded})))
        .mount(&server)
        .await;
    mount_invoke(
        &server,
        ResponseTemplate::new(200).set_body_json(json!({"data": "6869"})),
    )
    .await;
    let backend = backend_for(&server, &[]);
    assert_eq!(backend.read_bytes("/x").await.expect("read"), b"hi");

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(CREATE_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": {}})))
        .mount(&server)
        .await;
    let backend = backend_for(&server, &[]);
    let error = backend.read_bytes("/x").await.expect_err("error");
    assert!(
        matches!(error, RemoteError::Transport(ref message) if message.contains("environmentId")),
        "{error:?}"
    );
}

#[test]
fn percent_encoding_keeps_only_unreserved_bytes() {
    assert_eq!(percent_encode_path("env-1.A_b~c"), "env-1.A_b~c");
    assert_eq!(percent_encode_path("a/b c"), "a%2Fb%20c");
    assert_eq!(percent_encode_path("é"), "%C3%A9");
}

#[tokio::test]
async fn invoke_percent_encodes_the_id_and_reuses_the_cached_environment() {
    let server = MockServer::start().await;
    mount_create(&server, "env/1").await;
    mount_invoke(
        &server,
        ResponseTemplate::new(200).set_body_json(json!({"data": "6869"})),
    )
    .await;
    let backend = backend_for(&server, &[]);
    assert_eq!(backend.read_bytes("/a").await.expect("first read"), b"hi");
    assert_eq!(backend.read_bytes("/b").await.expect("second read"), b"hi");

    let requests = server.received_requests().await.expect("requests");
    let invoke = requests
        .iter()
        .find(|request| request.url.path().contains("/invoke"))
        .expect("invoke request");
    assert_eq!(invoke.url.path(), "/direct/env%2F1/invoke");
    assert_eq!(
        invoke
            .headers
            .get("accept")
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let body = request_body(invoke);
    assert_eq!(body["action"], json!("file.read"));
    assert_eq!(body["args"], json!({"path": "/a", "binary": true}));
    assert_eq!(
        requests
            .iter()
            .filter(|request| request.url.path() == CREATE_PATH)
            .count(),
        1
    );
}

#[tokio::test]
async fn file_read_decodes_hex_base64_and_reports_errors() {
    let server = MockServer::start().await;
    mount_create(&server, "env-1").await;
    mount_invoke(&server, |request: &WireRequest| {
        let body = request_body(request);
        match body["args"]["path"].as_str().unwrap_or_default() {
            "/hex" => ResponseTemplate::new(200).set_body_json(json!({"data": "6869"})),
            "/content" => ResponseTemplate::new(200).set_body_json(json!({"content": "6869"})),
            "/base64" => ResponseTemplate::new(200)
                .set_body_json(json!({"data": "aGk=", "content_encoding": "base64"})),
            "/missing" => ResponseTemplate::new(200)
                .set_body_json(json!({"error": "No such file or directory"})),
            _ => ResponseTemplate::new(200).set_body_json(json!({"error": "permission denied"})),
        }
    })
    .await;
    let backend = backend_for(&server, &[]);
    assert_eq!(backend.read_bytes("/hex").await.expect("hex"), b"hi");
    assert_eq!(
        backend.read_bytes("/content").await.expect("content"),
        b"hi"
    );
    assert_eq!(backend.read_bytes("/base64").await.expect("base64"), b"hi");

    let missing = backend.read_bytes("/missing").await.expect_err("missing");
    assert!(missing.is_not_found(), "{missing:?}");
    let denied = backend.read_bytes("/denied").await.expect_err("denied");
    assert!(
        matches!(denied, RemoteError::Status { status: 0, .. }),
        "{denied:?}"
    );
}

#[tokio::test]
async fn file_write_sends_binary_hex() {
    let server = MockServer::start().await;
    mount_create(&server, "env-1").await;
    mount_invoke(
        &server,
        ResponseTemplate::new(200).set_body_json(json!({"data": {}})),
    )
    .await;
    let backend = backend_for(&server, &[]);
    backend
        .write_bytes("/tmp/out.bin", b"\x00hi", true)
        .await
        .expect("write");

    let bodies = action_bodies(&server, "file.write").await;
    assert_eq!(
        bodies[0]["args"],
        json!({
            "path": "/tmp/out.bin",
            "data": "006869",
            "binary": true,
            "append": true,
        })
    );
}

#[tokio::test]
async fn file_list_maps_depth_items_and_filters() {
    let server = MockServer::start().await;
    mount_create(&server, "env-1").await;
    mount_invoke(
        &server,
        ResponseTemplate::new(200).set_body_json(json!({
            "entries": [
                {
                    "name": "a.txt",
                    "path": "/d/a.txt",
                    "size": 3,
                    "type": "file",
                    "modified_time": 1_695_441_203.5,
                },
                {"name": "sub", "path": "/d/sub", "type": "directory"},
                {"name": "weird", "path": "/d/weird", "type": ".log", "modified_time": null},
                {"name": "flag", "path": "/d/flag", "is_directory": true, "modified_time": true},
            ]
        })),
    )
    .await;
    let backend = backend_for(&server, &[]);

    let entries = backend
        .list(&ListRequest {
            path: "/d".to_string(),
            recursive: false,
            max_depth: None,
            include_files: true,
            include_dirs: true,
        })
        .await
        .expect("list");
    assert_eq!(entries.len(), 4);
    assert_eq!(entries[0].name, "a.txt");
    assert_eq!(entries[0].path, "/d/a.txt");
    assert_eq!(entries[0].size, 3);
    assert!(!entries[0].is_directory);
    assert_eq!(entries[0].modified_time, "1695441203.5");
    assert_eq!(entries[0].file_type, None, "'file' is a category type");
    assert!(entries[1].is_directory);
    assert_eq!(entries[1].file_type, None);
    assert_eq!(entries[2].file_type, Some(".log".to_string()));
    assert_eq!(entries[2].modified_time, "0");
    assert!(entries[3].is_directory);
    assert_eq!(entries[3].modified_time, "0");

    let files = backend
        .list(&ListRequest {
            path: "/d".to_string(),
            recursive: false,
            max_depth: None,
            include_files: true,
            include_dirs: false,
        })
        .await
        .expect("files");
    assert_eq!(files.len(), 2);
    let dirs = backend
        .list(&ListRequest {
            path: "/d".to_string(),
            recursive: false,
            max_depth: None,
            include_files: false,
            include_dirs: true,
        })
        .await
        .expect("dirs");
    assert_eq!(dirs.len(), 2);

    let _ = backend
        .list(&ListRequest {
            path: "/d".to_string(),
            recursive: true,
            max_depth: None,
            include_files: true,
            include_dirs: true,
        })
        .await
        .expect("unlimited recursion");
    let _ = backend
        .list(&ListRequest {
            path: "/d".to_string(),
            recursive: true,
            max_depth: Some(1),
            include_files: true,
            include_dirs: true,
        })
        .await
        .expect("bounded recursion");

    let depths: Vec<u64> = action_bodies(&server, "file.list")
        .await
        .iter()
        .map(|body| body["args"]["depth"].as_u64().expect("depth"))
        .collect();
    assert_eq!(depths, vec![1, 1, 1, 21, 2]);
}

#[tokio::test]
async fn file_search_sends_patterns_and_maps_the_limit_error() {
    let server = MockServer::start().await;
    mount_create(&server, "env-1").await;
    mount_invoke(
        &server,
        ResponseTemplate::new(200).set_body_json(json!({
            "items": [{"name": "main.rs", "path": "/src/main.rs", "type": ".rs"}]
        })),
    )
    .await;
    let backend = backend_for(&server, &[]);
    let entries = backend
        .search(&SearchRequest {
            path: "/src".to_string(),
            pattern: "*.rs".to_string(),
            exclude_patterns: Some(vec!["target/**".to_string()]),
        })
        .await
        .expect("search");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].file_type, Some(".rs".to_string()));
    let bodies = action_bodies(&server, "file.search").await;
    assert_eq!(
        bodies[0]["args"],
        json!({"path": "/src", "pattern": "*.rs", "exclude_patterns": ["target/**"]})
    );

    let server = MockServer::start().await;
    mount_create(&server, "env-1").await;
    mount_invoke(
        &server,
        ResponseTemplate::new(429).set_body_json(json!({"error": "too many matches"})),
    )
    .await;
    let backend = backend_for(&server, &[]);
    let error = backend
        .search(&SearchRequest {
            path: "/src".to_string(),
            pattern: "*".to_string(),
            exclude_patterns: None,
        })
        .await
        .expect_err("limit error");
    assert!(
        matches!(error, RemoteError::SearchLimit(ref message) if message == "too many matches"),
        "{error:?}"
    );

    let server = MockServer::start().await;
    mount_create(&server, "env-1").await;
    mount_invoke(&server, ResponseTemplate::new(429)).await;
    let backend = backend_for(&server, &[]);
    let error = backend
        .search(&SearchRequest {
            path: "/src".to_string(),
            pattern: "*".to_string(),
            exclude_patterns: None,
        })
        .await
        .expect_err("limit error");
    assert!(
        matches!(error, RemoteError::SearchLimit(ref message) if message == SEARCH_LIMIT_MESSAGE),
        "{error:?}"
    );
}

#[tokio::test]
async fn execute_folds_the_environment_and_normalizes_results() {
    let server = MockServer::start().await;
    mount_create(&server, "env-1").await;
    mount_invoke(&server, |request: &WireRequest| {
        let body = request_body(request);
        match body["args"]["cmd"].as_str().unwrap_or_default() {
            "export FOO=bar && echo hi" => ResponseTemplate::new(200)
                .set_body_json(json!({"returncode": 3, "stdout": "out\n", "stderr": ""})),
            "unset -- DROP && echo hi" => {
                ResponseTemplate::new(200).set_body_json(json!({"exit_code": 0, "stdout": "ok"}))
            }
            "echo hi" => ResponseTemplate::new(200).set_body_json(json!({"exit_code": 124})),
            "slow" => ResponseTemplate::new(200)
                .set_body_json(json!({"exit_code": 1, "stderr": "Command TIMED OUT"})),
            _ => ResponseTemplate::new(200).set_body_json(json!("oops")),
        }
    })
    .await;
    let backend = backend_for(&server, &[]);

    let mut environment = BTreeMap::new();
    environment.insert("FOO".to_string(), Some("bar".to_string()));
    let result = backend
        .execute(&ExecRequest {
            command: ExecCommand::Shell("echo hi".to_string()),
            cwd: Some(".".to_string()),
            environment: environment.clone(),
            stdin: None,
            timeout_secs: Some(0),
        })
        .await
        .expect("exec");
    assert_eq!(result.exit_code, 3);
    assert_eq!(result.stdout, "out\n");
    assert!(!result.timed_out);

    let mut environment = BTreeMap::new();
    environment.insert("DROP".to_string(), None);
    let result = backend
        .execute(&ExecRequest {
            command: ExecCommand::Shell("echo hi".to_string()),
            cwd: None,
            environment,
            stdin: None,
            timeout_secs: None,
        })
        .await
        .expect("exec");
    assert_eq!(result.exit_code, 0);

    let result = backend
        .execute(&ExecRequest {
            command: ExecCommand::Shell("echo hi".to_string()),
            cwd: Some("/tmp".to_string()),
            environment: BTreeMap::new(),
            stdin: None,
            timeout_secs: Some(5),
        })
        .await
        .expect("exec");
    assert_eq!(result.exit_code, 124);
    assert!(result.timed_out);

    let result = backend
        .execute(&ExecRequest {
            command: ExecCommand::Shell("slow".to_string()),
            cwd: None,
            environment: BTreeMap::new(),
            stdin: None,
            timeout_secs: None,
        })
        .await
        .expect("exec");
    assert!(result.timed_out);
    assert_eq!(result.stderr, "Command TIMED OUT");

    let result = backend
        .execute(&ExecRequest {
            command: ExecCommand::Shell("other".to_string()),
            cwd: None,
            environment: BTreeMap::new(),
            stdin: None,
            timeout_secs: None,
        })
        .await
        .expect("exec");
    assert_eq!(result.exit_code, -1);
    assert_eq!(result.stderr, "unexpected result type");

    let bodies = action_bodies(&server, "process.exec").await;
    assert_eq!(
        bodies[0]["args"],
        json!({"cmd": "export FOO=bar && echo hi"})
    );
    assert_eq!(
        bodies[1]["args"],
        json!({"cmd": "unset -- DROP && echo hi"})
    );
    assert_eq!(
        bodies[2]["args"],
        json!({"cmd": "echo hi", "cwd": "/tmp", "timeout": 5})
    );
}

#[test]
fn code_commands_wrap_the_snippet_in_base64() {
    let backend = YuanRongBackend::new(&endpoint("http://127.0.0.1:9"), &gateway_config(&[]))
        .expect("backend");
    let encoded = base64_encode(b"print('hi')");
    let ExecCommand::Shell(python) = backend.code_command("print('hi')", Language::Python, false)
    else {
        panic!("python must run through the shell");
    };
    assert!(python.starts_with("python3 -c "), "{python}");
    assert!(python.contains(&encoded), "{python}");
    assert!(python.contains("base64.b64decode"), "{python}");

    let ExecCommand::Shell(javascript) =
        backend.code_command("console.log(1)", Language::Javascript, false)
    else {
        panic!("javascript must run through the shell");
    };
    assert!(javascript.starts_with("node -e "), "{javascript}");
    assert!(javascript.contains("Buffer.from"), "{javascript}");

    let ExecCommand::Shell(file) = backend.code_command("print(1)", Language::Python, true) else {
        panic!("a forced file must run through the shell");
    };
    assert!(file.contains("mktemp /tmp/ojw_code_XXXXXX.py"), "{file}");
    assert!(file.contains("base64 -d"), "{file}");
    assert!(file.contains("exit $status"), "{file}");

    assert_eq!(backend.code_cwd(Some("/custom")), "/tmp");
}

#[tokio::test]
async fn invoke_404_recreates_once_and_deletes_the_stale_environment() {
    let server = MockServer::start().await;
    let creates = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&creates);
    Mock::given(method("POST"))
        .and(path(CREATE_PATH))
        .respond_with(move |_request: &WireRequest| {
            let index = counter.fetch_add(1, Ordering::SeqCst);
            let environment_id = if index == 0 { "env-1" } else { "env-2" };
            ResponseTemplate::new(200)
                .set_body_json(json!({"data": {"environmentId": environment_id}}))
        })
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/direct/env-1/invoke"))
        .respond_with(
            ResponseTemplate::new(404).set_body_json(json!({"error": "environment not found"})),
        )
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/direct/env-2/invoke"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": "6869"})))
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/api/environment/v1/environments/env-1"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    let backend = backend_for(&server, &[]).with_recreate(1, Duration::ZERO);
    assert_eq!(backend.read_bytes("/x").await.expect("read"), b"hi");
    assert_eq!(creates.load(Ordering::SeqCst), 2);

    let requests = server.received_requests().await.expect("requests");
    assert_eq!(
        requests
            .iter()
            .filter(|request| request.url.path() == CREATE_PATH)
            .count(),
        2
    );
    assert!(requests.iter().any(|request| {
        request.method.as_str() == "DELETE"
            && request.url.path() == "/api/environment/v1/environments/env-1"
    }));
}

#[tokio::test]
async fn exhausted_recreates_surface_container_gone() {
    let server = MockServer::start().await;
    mount_create(&server, "env-1").await;
    mount_invoke(
        &server,
        ResponseTemplate::new(404).set_body_json(json!({"error": "gone"})),
    )
    .await;
    Mock::given(method("DELETE"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    let backend = backend_for(&server, &[]).with_recreate(0, Duration::ZERO);
    let error = backend.read_bytes("/x").await.expect_err("error");
    assert!(error.is_container_gone(), "{error:?}");
    assert_eq!(error.message(), "HTTP 404 for file.read: gone");

    let requests = server.received_requests().await.expect("requests");
    assert!(requests.iter().any(|request| {
        request.method.as_str() == "DELETE"
            && request.url.path() == "/api/environment/v1/environments/env-1"
    }));
}

#[test]
fn provider_set_registers_three_factories_and_requires_a_base_url() {
    let set = provider_set();
    assert!(!set.is_empty());
    assert!(set.fs.is_some());
    assert!(set.shell.is_some());
    assert!(set.code.is_some());

    let error = YuanRongBackend::new(&endpoint("   "), &SandboxGatewayConfig::default())
        .err()
        .expect("error");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(
        error.message(),
        "yuanrong provider requires endpoint.base_url"
    );

    let factory = set.fs.as_ref().expect("fs factory");
    let error = factory(&endpoint("  "), &SandboxGatewayConfig::default())
        .err()
        .expect("factory error");
    assert_eq!(error.kind(), ErrorKind::Config);
}
