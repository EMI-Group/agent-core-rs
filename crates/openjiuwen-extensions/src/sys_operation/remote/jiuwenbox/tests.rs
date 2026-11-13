//! Tests for the `jiuwenbox` wire backend against a loopback mock server.
//!
//! Python counterparts: `tests/unit_tests/extensions/sys_operation/sandbox/`
//! (the `jiuwenbox` provider suites) — the request shapes, the create options,
//! the idle-timeout dedupe, the sandbox-not-found recreate, and the tolerant
//! listing mapping are asserted here against `wiremock` instead of `respx`.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_core::sys_operation::sandbox::config::{
    SandboxEndpoint, SandboxGatewayConfig, SandboxLauncherConfig,
};
use serde_json::{json, Value};
use wiremock::matchers::{
    body_json, body_string_contains, header, method, path, query_param, query_param_is_missing,
};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::*;

/// The sandbox id every create mock hands out.
const SANDBOX_ID: &str = "sb-1";

/// No recreate attempts and no delay: call counts stay predictable.
const NO_RECREATE: (u32, Duration) = (0, Duration::ZERO);

fn endpoint(server: &MockServer) -> SandboxEndpoint {
    SandboxEndpoint {
        base_url: server.uri(),
        sandbox_id: None,
        isolation_key: Some(unique_isolation_key()),
    }
}

/// Each test uses its own isolation key, so the process-global container cache
/// cannot hand an id to a test whose loopback port was recycled.
fn unique_isolation_key() -> String {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    format!("test-{}", NEXT.fetch_add(1, Ordering::Relaxed))
}

fn backend(server: &MockServer) -> JiuwenBoxBackend {
    JiuwenBoxBackend::new(&endpoint(server), &SandboxGatewayConfig::default())
        .unwrap()
        .with_recreate(NO_RECREATE.0, NO_RECREATE.1)
}

/// A configuration whose launcher carries `extra`.
fn config_with_extra(extra: &Value) -> SandboxGatewayConfig {
    let extra_params = extra
        .as_object()
        .unwrap()
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    SandboxGatewayConfig {
        launcher_config: Some(SandboxLauncherConfig {
            extra_params,
            ..SandboxLauncherConfig::default()
        }),
        ..SandboxGatewayConfig::default()
    }
}

fn sandbox_path(suffix: &str) -> String {
    format!("/api/v1/sandboxes/{SANDBOX_ID}/{suffix}")
}

async fn mount_create(server: &MockServer, sandbox_id: &str) {
    Mock::given(method("POST"))
        .and(path("/api/v1/sandboxes"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "id": sandbox_id })))
        .mount(server)
        .await;
}

async fn mount_download(server: &MockServer, body: &'static [u8]) {
    Mock::given(method("GET"))
        .and(path(sandbox_path("download")))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(body.to_vec()))
        .mount(server)
        .await;
}

// --- endpoint and configuration -------------------------------------------

#[tokio::test]
async fn a_bare_host_and_port_is_reached_over_http() {
    let server = MockServer::start().await;
    mount_create(&server, SANDBOX_ID).await;
    Mock::given(method("GET"))
        .and(path(sandbox_path("download")))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"hello".to_vec()))
        .expect(1)
        .mount(&server)
        .await;
    let endpoint = SandboxEndpoint {
        base_url: format!("127.0.0.1:{}/", server.address().port()),
        sandbox_id: None,
        isolation_key: Some(unique_isolation_key()),
    };
    let backend = JiuwenBoxBackend::new(&endpoint, &SandboxGatewayConfig::default()).unwrap();

    assert_eq!(
        backend.http.base_url(),
        format!("http://127.0.0.1:{}", server.address().port())
    );
    assert_eq!(backend.read_bytes("/etc/hostname").await.unwrap(), b"hello");
}

#[test]
fn a_missing_base_url_is_a_configuration_error() {
    let endpoint = SandboxEndpoint {
        base_url: "   ".to_string(),
        sandbox_id: None,
        isolation_key: None,
    };
    let error = JiuwenBoxBackend::new(&endpoint, &SandboxGatewayConfig::default())
        .err()
        .unwrap();

    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(
        error.message(),
        "jiuwenbox provider requires endpoint.base_url"
    );
}

// --- container creation ----------------------------------------------------

#[tokio::test]
async fn the_create_body_passes_the_launcher_options_through() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/sandboxes"))
        .and(body_json(json!({
            "policy": {"mode": "allow"},
            "policy_mode": "enforce",
            "sandbox_runtime": "conch",
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "id": SANDBOX_ID })))
        .expect(1)
        .mount(&server)
        .await;
    mount_download(&server, b"x").await;
    // `api_sandbox_runtime` wins over `sandbox_runtime`, as in Python's Swarm mapping.
    let config = config_with_extra(&json!({
        "policy": {"mode": "allow"},
        "policy_mode": "enforce",
        "api_sandbox_runtime": "conch",
        "sandbox_runtime": "python",
    }));
    let backend = JiuwenBoxBackend::new(&endpoint(&server), &config)
        .unwrap()
        .with_recreate(NO_RECREATE.0, NO_RECREATE.1);

    assert_eq!(backend.read_bytes("/a").await.unwrap(), b"x");
}

#[tokio::test]
async fn the_create_body_is_empty_without_launcher_options() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/sandboxes"))
        .and(body_json(json!({})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "id": SANDBOX_ID })))
        .expect(1)
        .mount(&server)
        .await;
    mount_download(&server, b"x").await;

    assert_eq!(backend(&server).read_bytes("/a").await.unwrap(), b"x");
}

#[tokio::test]
async fn a_falsy_api_sandbox_runtime_falls_back_to_the_plain_option() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/sandboxes"))
        .and(body_json(json!({ "sandbox_runtime": "python" })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "id": SANDBOX_ID })))
        .expect(1)
        .mount(&server)
        .await;
    mount_download(&server, b"x").await;
    let config = config_with_extra(&json!({
        "api_sandbox_runtime": "",
        "sandbox_runtime": "  python  ",
    }));
    let backend = JiuwenBoxBackend::new(&endpoint(&server), &config)
        .unwrap()
        .with_recreate(NO_RECREATE.0, NO_RECREATE.1);

    assert_eq!(backend.read_bytes("/a").await.unwrap(), b"x");
}

#[tokio::test]
async fn the_created_id_is_reused_across_operations() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/sandboxes"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "id": SANDBOX_ID })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path(sandbox_path("download")))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"x".to_vec()))
        .expect(2)
        .mount(&server)
        .await;
    let backend = backend(&server);

    backend.read_bytes("/a").await.unwrap();
    backend.read_bytes("/b").await.unwrap();
}

#[tokio::test]
async fn a_create_answer_without_an_id_is_a_transport_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/sandboxes"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "id": "  " })))
        .expect(1)
        .mount(&server)
        .await;
    let error = backend(&server).read_bytes("/a").await.unwrap_err();

    assert!(matches!(error, RemoteError::Transport(_)));
    assert!(error.message().contains("returned no sandbox id"));
}

#[tokio::test]
async fn the_idle_policy_is_written_once_per_base_url() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/sandboxes"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "id": SANDBOX_ID })))
        .expect(2)
        .mount(&server)
        .await;
    Mock::given(method("PUT"))
        .and(path("/api/v1/timeout"))
        .and(body_json(
            json!({ "idle_timeout": 60, "idle_check_interval": 5 }),
        ))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;
    mount_download(&server, b"x").await;
    let config = SandboxGatewayConfig {
        launcher_config: Some(SandboxLauncherConfig {
            idle_ttl_seconds: Some(60),
            extra_params: [(String::from("idle_check_interval"), json!(5))]
                .into_iter()
                .collect(),
            ..SandboxLauncherConfig::default()
        }),
        ..SandboxGatewayConfig::default()
    };

    // Two isolation keys on one base URL share the idle policy but not the container.
    for isolation_key in ["agent-a", "agent-b"] {
        let endpoint = SandboxEndpoint {
            base_url: server.uri(),
            sandbox_id: None,
            isolation_key: Some(isolation_key.to_string()),
        };
        let backend = JiuwenBoxBackend::new(&endpoint, &config)
            .unwrap()
            .with_recreate(NO_RECREATE.0, NO_RECREATE.1);
        assert_eq!(backend.read_bytes("/a").await.unwrap(), b"x");
    }
}

#[tokio::test]
async fn no_idle_policy_is_written_without_launcher_settings() {
    let server = MockServer::start().await;
    mount_create(&server, SANDBOX_ID).await;
    mount_download(&server, b"x").await;

    backend(&server).read_bytes("/a").await.unwrap();

    let requests = server.received_requests().await.unwrap();
    assert!(
        requests
            .iter()
            .all(|request| request.method != reqwest::Method::PUT),
        "an unconfigured idle policy must not reach the server"
    );
}

// --- exec ------------------------------------------------------------------

#[tokio::test]
async fn exec_omits_unset_fields_and_keeps_null_environment_values() {
    let server = MockServer::start().await;
    mount_create(&server, SANDBOX_ID).await;
    Mock::given(method("POST"))
        .and(path(sandbox_path("exec")))
        .and(body_json(json!({
            "command": ["bash", "-lc", "ls -la"],
            "env": {"KEEP": "1", "DROP": null},
            "stdin": "input",
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({ "exit_code": 0, "stdout": "ok", "stderr": "" })),
        )
        .expect(1)
        .mount(&server)
        .await;
    let backend = backend(&server);
    let mut request = ExecRequest::new(ExecCommand::Shell("ls -la".to_string()));
    request.cwd = Some(".".to_string());
    request
        .environment
        .insert("KEEP".to_string(), Some("1".to_string()));
    request.environment.insert("DROP".to_string(), None);
    request.stdin = Some("input".to_string());
    request.timeout_secs = Some(0);

    let result = backend.execute(&request).await.unwrap();

    assert_eq!(
        result,
        ExecResult {
            exit_code: 0,
            stdout: "ok".to_string(),
            stderr: String::new(),
            timed_out: false,
        }
    );
}

#[tokio::test]
async fn exec_sends_argv_verbatim_with_a_workdir_and_timeout() {
    let server = MockServer::start().await;
    mount_create(&server, SANDBOX_ID).await;
    Mock::given(method("POST"))
        .and(path(sandbox_path("exec")))
        .and(body_json(json!({
            "command": ["echo", "hi"],
            "workdir": "/tmp",
            "timeout_seconds": 7,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "stdout": "hi\n" })))
        .expect(1)
        .mount(&server)
        .await;
    let backend = backend(&server);
    let mut request = ExecRequest::new(ExecCommand::argv(["echo", "hi"]));
    request.cwd = Some("/tmp".to_string());
    request.timeout_secs = Some(7);

    let result = backend.execute(&request).await.unwrap();

    assert_eq!(result.exit_code, 0, "a missing exit_code means success");
    assert_eq!(result.stdout, "hi\n");
    assert!(!result.timed_out);
}

#[tokio::test]
async fn only_exit_124_counts_as_a_timeout() {
    let server = MockServer::start().await;
    mount_create(&server, SANDBOX_ID).await;
    Mock::given(method("POST"))
        .and(path(sandbox_path("exec")))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({ "exit_code": 124, "stderr": "killed" })),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(sandbox_path("exec")))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({ "exit_code": 2, "stderr": "bad flag" })),
        )
        .mount(&server)
        .await;
    let backend = backend(&server);

    let timed_out = backend
        .execute(&ExecRequest::new(ExecCommand::Shell(
            "sleep 99".to_string(),
        )))
        .await
        .unwrap();
    assert_eq!(timed_out.exit_code, 124);
    assert!(timed_out.timed_out);

    let failed = backend
        .execute(&ExecRequest::new(ExecCommand::Shell(
            "ls --bad".to_string(),
        )))
        .await
        .unwrap();
    assert_eq!(failed.exit_code, 2);
    assert!(!failed.timed_out, "a non-zero exit is a normal result");
    assert_eq!(failed.stderr, "bad flag");
}

// --- transfers -------------------------------------------------------------

#[tokio::test]
async fn reading_downloads_with_the_sandbox_path_query() {
    let server = MockServer::start().await;
    mount_create(&server, SANDBOX_ID).await;
    Mock::given(method("GET"))
        .and(path(sandbox_path("download")))
        .and(query_param("sandbox_path", "/tmp/a.txt"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"data".to_vec()))
        .expect(1)
        .mount(&server)
        .await;

    let bytes = backend(&server).read_bytes("/tmp/a.txt").await.unwrap();

    assert_eq!(bytes, b"data");
}

#[tokio::test]
async fn overwriting_uploads_one_multipart_file() {
    let server = MockServer::start().await;
    mount_create(&server, SANDBOX_ID).await;
    Mock::given(method("POST"))
        .and(path(sandbox_path("upload")))
        .and(query_param("sandbox_path", "/tmp/report.txt"))
        .and(body_string_contains("name=\"file\""))
        .and(body_string_contains("filename=\"report.txt\""))
        .and(body_string_contains("payload"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;

    backend(&server)
        .write_bytes("/tmp/report.txt", b"payload", false)
        .await
        .unwrap();
}

#[tokio::test]
async fn appending_runs_the_base64_script_with_stdin() {
    let server = MockServer::start().await;
    mount_create(&server, SANDBOX_ID).await;
    Mock::given(method("POST"))
        .and(path(sandbox_path("exec")))
        .and(body_json(json!({
            "command": ["bash", "-lc", APPEND_SCRIPT, "jiuwenbox-append", "/tmp/log.txt"],
            "stdin": base64_encode(b"more"),
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "exit_code": 0 })))
        .expect(1)
        .mount(&server)
        .await;

    backend(&server)
        .write_bytes("/tmp/log.txt", b"more", true)
        .await
        .unwrap();
}

#[tokio::test]
async fn a_failed_append_reports_the_container_error() {
    let server = MockServer::start().await;
    mount_create(&server, SANDBOX_ID).await;
    Mock::given(method("POST"))
        .and(path(sandbox_path("exec")))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({ "exit_code": 1, "stderr": "disk full" })),
        )
        .expect(1)
        .mount(&server)
        .await;

    let error = backend(&server)
        .write_bytes("/tmp/log.txt", b"more", true)
        .await
        .unwrap_err();

    match error {
        RemoteError::Status { status: 0, message } => assert_eq!(message, "disk full"),
        other => panic!("unexpected error: {}", other.message()),
    }
}

// --- listings and search ---------------------------------------------------

#[tokio::test]
async fn listing_sends_every_filter_and_maps_tolerant_items() {
    let server = MockServer::start().await;
    mount_create(&server, SANDBOX_ID).await;
    Mock::given(method("GET"))
        .and(path(sandbox_path("files")))
        .and(query_param("sandbox_path", "/tmp"))
        .and(query_param("recursive", "true"))
        .and(query_param("include_files", "true"))
        .and(query_param("include_dirs", "false"))
        .and(query_param("max_depth", "2"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "items": [
                {
                    "name": "a.txt",
                    "path": "/tmp/a.txt",
                    "size": 12,
                    "is_directory": false,
                    "modified_time": "2024-01-01 00:00:00",
                    "type": ".txt",
                },
                { "name": "sub" },
            ],
        })))
        .expect(1)
        .mount(&server)
        .await;
    let request = ListRequest {
        path: "/tmp".to_string(),
        recursive: true,
        max_depth: Some(2),
        include_files: true,
        include_dirs: false,
    };

    let entries = backend(&server).list(&request).await.unwrap();

    assert_eq!(entries.len(), 2);
    assert_eq!(
        entries[0],
        RemoteEntry {
            name: "a.txt".to_string(),
            path: "/tmp/a.txt".to_string(),
            size: 12,
            modified_time: "2024-01-01 00:00:00".to_string(),
            is_directory: false,
            file_type: Some(".txt".to_string()),
        }
    );
    assert_eq!(
        entries[1],
        RemoteEntry {
            name: "sub".to_string(),
            path: String::new(),
            size: 0,
            modified_time: "0".to_string(),
            is_directory: false,
            file_type: None,
        },
        "a bare item keeps Python's defaults"
    );
}

#[tokio::test]
async fn a_listing_without_a_depth_limit_omits_the_parameter() {
    let server = MockServer::start().await;
    mount_create(&server, SANDBOX_ID).await;
    Mock::given(method("GET"))
        .and(path(sandbox_path("files")))
        .and(query_param_is_missing("max_depth"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "items": [] })))
        .expect(1)
        .mount(&server)
        .await;
    let request = ListRequest {
        path: "/tmp".to_string(),
        recursive: false,
        max_depth: None,
        include_files: true,
        include_dirs: true,
    };

    assert_eq!(backend(&server).list(&request).await.unwrap(), Vec::new());
}

#[tokio::test]
async fn searching_repeats_the_exclude_patterns() {
    let server = MockServer::start().await;
    mount_create(&server, SANDBOX_ID).await;
    Mock::given(method("GET"))
        .and(path(sandbox_path("search")))
        .and(query_param("sandbox_path", "/tmp"))
        .and(query_param("pattern", "*.rs"))
        .and(query_param("exclude_patterns", "*.log"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "items": [{ "name": "main.rs", "path": "/tmp/main.rs", "type": ".rs" }],
        })))
        .expect(1)
        .mount(&server)
        .await;
    let request = SearchRequest {
        path: "/tmp".to_string(),
        pattern: "*.rs".to_string(),
        exclude_patterns: Some(vec!["*.log".to_string(), "*.tmp".to_string()]),
    };

    let entries = backend(&server).search(&request).await.unwrap();

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].name, "main.rs");
    let requests = server.received_requests().await.unwrap();
    let search = requests
        .iter()
        .find(|received| received.url.path() == sandbox_path("search"))
        .unwrap();
    let query = search
        .url
        .query_pairs()
        .filter(|(key, _)| key == "exclude_patterns")
        .map(|(_, value)| value.to_string())
        .collect::<Vec<_>>();
    assert_eq!(query, vec!["*.log".to_string(), "*.tmp".to_string()]);
}

// --- recreate on a lost container -----------------------------------------

#[tokio::test]
async fn a_lost_container_is_recreated_once() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/sandboxes"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "id": SANDBOX_ID })))
        .expect(2)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path(sandbox_path("download")))
        .respond_with(
            ResponseTemplate::new(404)
                .set_body_json(json!({ "error": "Sandbox 'sb-1' not found" })),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path(sandbox_path("download")))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"recovered".to_vec()))
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path(format!("/api/v1/sandboxes/{SANDBOX_ID}")))
        .respond_with(ResponseTemplate::new(404))
        .expect(1)
        .mount(&server)
        .await;
    let backend = JiuwenBoxBackend::new(&endpoint(&server), &SandboxGatewayConfig::default())
        .unwrap()
        .with_recreate(1, Duration::ZERO);

    assert_eq!(backend.read_bytes("/a").await.unwrap(), b"recovered");
}

#[tokio::test]
async fn a_file_level_404_is_not_a_lost_container() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/sandboxes"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "id": SANDBOX_ID })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path(sandbox_path("download")))
        .respond_with(
            ResponseTemplate::new(404).set_body_json(json!({ "detail": "/nope: file not found" })),
        )
        .expect(1)
        .mount(&server)
        .await;
    let backend = JiuwenBoxBackend::new(&endpoint(&server), &SandboxGatewayConfig::default())
        .unwrap()
        .with_recreate(2, Duration::ZERO);

    let error = backend.read_bytes("/nope").await.unwrap_err();

    assert!(error.is_not_found());
    assert_eq!(
        error.message(),
        "HTTP 404 for read_file: /nope: file not found"
    );
}

#[tokio::test]
async fn exhausted_recreates_surface_the_container_gone_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/sandboxes"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "id": SANDBOX_ID })))
        .expect(2)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path(sandbox_path("download")))
        .respond_with(
            ResponseTemplate::new(404)
                .set_body_json(json!({ "message": "Sandbox 'sb-1' not found" })),
        )
        .expect(2)
        .mount(&server)
        .await;
    let backend = JiuwenBoxBackend::new(&endpoint(&server), &SandboxGatewayConfig::default())
        .unwrap()
        .with_recreate(1, Duration::ZERO);

    let error = backend.read_bytes("/a").await.unwrap_err();

    assert!(error.is_container_gone());
}

// --- auth and failures -----------------------------------------------------

#[tokio::test]
async fn configured_auth_headers_and_query_parameters_are_sent() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/sandboxes"))
        .and(header("x-api-key", "k-1"))
        .and(query_param("tenant", "t-1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "id": SANDBOX_ID })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path(sandbox_path("download")))
        .and(header("x-api-key", "k-1"))
        .and(query_param("tenant", "t-1"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"x".to_vec()))
        .expect(1)
        .mount(&server)
        .await;
    let mut config = SandboxGatewayConfig::default();
    config
        .auth_headers
        .insert("x-api-key".to_string(), "k-1".to_string());
    config
        .auth_query_params
        .insert("tenant".to_string(), "t-1".to_string());
    let backend = JiuwenBoxBackend::new(&endpoint(&server), &config)
        .unwrap()
        .with_recreate(NO_RECREATE.0, NO_RECREATE.1);

    assert_eq!(backend.read_bytes("/a").await.unwrap(), b"x");
}

#[tokio::test]
async fn the_bearer_token_is_only_sent_when_provided() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/sandboxes"))
        .and(header("authorization", "Bearer secret"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "id": SANDBOX_ID })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path(sandbox_path("download")))
        .and(header("authorization", "Bearer secret"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"x".to_vec()))
        .expect(1)
        .mount(&server)
        .await;
    let backend = JiuwenBoxBackend::new(&endpoint(&server), &SandboxGatewayConfig::default())
        .unwrap()
        .with_bearer_token(Some("secret".to_string()));

    assert_eq!(backend.read_bytes("/a").await.unwrap(), b"x");
}

#[tokio::test]
async fn no_bearer_token_is_invented() {
    let server = MockServer::start().await;
    mount_create(&server, SANDBOX_ID).await;
    mount_download(&server, b"x").await;
    let backend = JiuwenBoxBackend::new(&endpoint(&server), &SandboxGatewayConfig::default())
        .unwrap()
        .with_bearer_token(None);

    backend.read_bytes("/a").await.unwrap();

    let requests = server.received_requests().await.unwrap();
    assert!(
        requests
            .iter()
            .all(|request| !request.headers.contains_key("authorization")),
        "an absent token must not add an Authorization header"
    );
}

#[tokio::test]
async fn a_server_error_becomes_a_status_error() {
    let server = MockServer::start().await;
    mount_create(&server, SANDBOX_ID).await;
    Mock::given(method("GET"))
        .and(path(sandbox_path("download")))
        .respond_with(ResponseTemplate::new(500).set_body_json(json!({ "error": "boom" })))
        .expect(1)
        .mount(&server)
        .await;

    let error = backend(&server).read_bytes("/a").await.unwrap_err();

    assert!(matches!(error, RemoteError::Status { status: 500, .. }));
    assert_eq!(error.message(), "HTTP 500 for read_file: boom");
}

#[tokio::test]
async fn an_unreachable_server_is_a_transport_error() {
    let endpoint = SandboxEndpoint {
        base_url: closed_http_endpoint(),
        sandbox_id: None,
        isolation_key: Some(unique_isolation_key()),
    };

    let backend = JiuwenBoxBackend::new(&endpoint, &SandboxGatewayConfig::default())
        .unwrap()
        .with_recreate(1, Duration::ZERO);
    let error = backend.read_bytes("/a").await.unwrap_err();

    assert!(matches!(error, RemoteError::Transport(_)));
    assert!(!error.is_container_gone());
}

/// A loopback address that accepts and immediately closes connections, so every
/// request fails at the transport layer without waiting for a timeout.
fn closed_http_endpoint() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            drop(stream);
        }
    });
    format!("http://{address}")
}
