//! Conditional-write behavior across the crate boundary: `create_if_absent` and
//! `expected_content_sha256` honored by each remote provider over its own wire protocol.
//!
//! Python parity: the reference providers under
//! `openjiuwen/extensions/sys_operation/sandbox/providers/` implement neither option
//! (`openjiuwen-sysop`'s local provider is the only Python implementation), so these
//! cases pin the additive Rust capability described in
//! `crate::sys_operation::remote::conditional`: the precondition is evaluated host-side
//! over the provider's read and write primitives under a per-path in-process lock, and a
//! rejected write issues no write request at all.
//!
//! Every case runs against a loopback `wiremock` peer, with one isolation key per test and
//! every await bounded by [`BOUND`].

use std::fmt::Write as _;

use sha2::{Digest, Sha256};

use super::*;

/// Lower-case SHA-256 hex of `text`, the form `expected_content_sha256` carries.
fn sha256_hex(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .fold(String::new(), |mut out, byte| {
            write!(out, "{byte:02x}").expect("writing into a String cannot fail");
            out
        })
}

/// A `create_if_absent` write request; the kernel's default `create_if_not_exist`
/// satisfies the flag combination the preconditions accept.
fn create_if_absent(path: &str, content: &str) -> WriteFileParams {
    let mut params = WriteFileParams::raw(path, content);
    params.create_if_absent = true;
    params
}

/// A write request that only proceeds when the remote bytes hash to `digest`.
fn expecting(path: &str, content: &str, digest: &str) -> WriteFileParams {
    WriteFileParams::raw(path, content).with_expected_sha256(digest)
}

/// A gateway configuration whose compare-and-swap flag is `conditional`.
fn capability_config(base_url: &str, conditional: bool) -> SandboxGatewayConfig {
    SandboxGatewayConfig {
        timeout_seconds: 5,
        supports_conditional_write: conditional,
        launcher_config: Some(SandboxLauncherConfig {
            base_url: Some(base_url.to_string()),
            ..SandboxLauncherConfig::default()
        }),
        ..SandboxGatewayConfig::default()
    }
}

/// Every request of `server` whose `jiuwenbox` path ends with `suffix`.
async fn jiuwenbox_requests(server: &MockServer, suffix: &str) -> Vec<Request> {
    requests(server)
        .await
        .into_iter()
        .filter(|request| request.url.path().ends_with(suffix))
        .collect()
}

// ─── capability ─────────────────────────────────────────────────────────────
// Every provider evaluates the precondition itself, so the capability is the provider's
// and not the gateway configuration's (`SandboxGatewayConfig::supports_conditional_write`
// describes a gateway-side compare-and-swap primitive).
#[test]
fn every_provider_supports_conditional_writes_whatever_the_gateway_flag() {
    let registry = registry();
    for sandbox_type in REMOTE_SANDBOX_TYPES {
        for conditional in [false, true] {
            let config = capability_config("127.0.0.1:9", conditional);
            let endpoint = endpoint_at("127.0.0.1:9", None, "session_conditional_capability");
            let fs = fs_operation(&registry, sandbox_type, &endpoint, &config);
            assert!(
                fs.supports_conditional_write(),
                "{sandbox_type} must honour conditional writes when the gateway flag is {conditional}"
            );
        }
    }
}

// ─── jiuwenbox: native REST API ─────────────────────────────────────────────
/// Mounts the `jiuwenbox` read of `path`: `content` when it exists, a `404` when not.
async fn mount_jiuwenbox_read(server: &MockServer, remote_path: &str, content: Option<&str>) {
    let response = match content {
        Some(content) => ResponseTemplate::new(200).set_body_string(content),
        None => {
            ResponseTemplate::new(404).set_body_json(json!({"error": "no such file or directory"}))
        }
    };
    Mock::given(method("GET"))
        .and(path("/api/v1/sandboxes/sbx-conditional/download"))
        .and(query_param("sandbox_path", remote_path))
        .respond_with(response)
        .mount(server)
        .await;
}

/// Mounts the `jiuwenbox` upload endpoint (the only write request the provider issues).
async fn mount_jiuwenbox_upload(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/api/v1/sandboxes/sbx-conditional/upload"))
        .respond_with(ResponseTemplate::new(200))
        .mount(server)
        .await;
}

/// `create_if_absent` over the native REST API: a missing path uploads, an existing one is
/// refused after the read and before any write request.
#[tokio::test]
async fn jiuwenbox_create_if_absent_uploads_a_missing_path_and_refuses_an_existing_one() {
    let server = MockServer::start().await;
    mount_jiuwenbox_read(&server, "/workspace/new.txt", None).await;
    mount_jiuwenbox_read(&server, "/workspace/existing.txt", Some("original")).await;
    mount_jiuwenbox_upload(&server).await;

    let isolation = "session_jiuwenbox_create_if_absent";
    let config = jiuwenbox_config(&server, json!({"sandbox_id": "sbx-conditional"}));
    let fs = fs_operation(
        &registry(),
        "jiuwenbox",
        &endpoint(&server, None, isolation),
        &config,
    );

    let created = tokio::time::timeout(
        BOUND,
        fs.write_file(create_if_absent("/workspace/new.txt", "hello")),
    )
    .await
    .unwrap();
    assert_eq!(created.code, 0, "{}", created.message);
    let data = created.data.unwrap();
    assert_eq!(data.path, "/workspace/new.txt");
    assert_eq!(data.size, 5);

    let refused = tokio::time::timeout(
        BOUND,
        fs.write_file(create_if_absent("/workspace/existing.txt", "replacement")),
    )
    .await
    .unwrap();
    assert_eq!(refused.code, 199_003);
    assert!(
        refused.message.contains("/workspace/existing.txt"),
        "{}",
        refused.message
    );
    assert!(
        refused.message.contains("create_if_absent"),
        "{}",
        refused.message
    );

    // The read that precedes a conditional write is a `download`; only the missing path
    // then uploads.
    let reads = jiuwenbox_requests(&server, "/download").await;
    assert_eq!(reads.len(), 2, "each conditional write reads first");
    let uploads = jiuwenbox_requests(&server, "/upload").await;
    assert_eq!(uploads.len(), 1, "the refused write must not upload");
    assert!(has_query(&uploads[0], "sandbox_path", "/workspace/new.txt"));
    assert!(
        uploads.iter().all(|request| !has_query(
            request,
            "sandbox_path",
            "/workspace/existing.txt"
        )),
        "the existing path must never be uploaded"
    );
}

/// `expected_content_sha256` over the native REST API: a matching digest uploads, a
/// mismatch is refused with the expected and the actual digest, and no upload happens.
#[tokio::test]
async fn jiuwenbox_expected_sha256_compares_the_remote_bytes_before_uploading() {
    let server = MockServer::start().await;
    mount_jiuwenbox_read(&server, "/workspace/match.txt", Some("hello")).await;
    mount_jiuwenbox_read(&server, "/workspace/mismatch.txt", Some("hello")).await;
    mount_jiuwenbox_upload(&server).await;

    let isolation = "session_jiuwenbox_expected_sha256";
    let config = jiuwenbox_config(&server, json!({"sandbox_id": "sbx-conditional"}));
    let fs = fs_operation(
        &registry(),
        "jiuwenbox",
        &endpoint(&server, None, isolation),
        &config,
    );
    let hello = sha256_hex("hello");

    let matched = tokio::time::timeout(
        BOUND,
        fs.write_file(expecting("/workspace/match.txt", "goodbye", &hello)),
    )
    .await
    .unwrap();
    assert_eq!(matched.code, 0, "{}", matched.message);

    let expected = sha256_hex("goodbye");
    let refused = tokio::time::timeout(
        BOUND,
        fs.write_file(expecting(
            "/workspace/mismatch.txt",
            "replacement",
            &expected,
        )),
    )
    .await
    .unwrap();
    assert_eq!(refused.code, 199_003);
    assert!(
        refused.message.contains("/workspace/mismatch.txt"),
        "{}",
        refused.message
    );
    assert!(
        refused.message.contains(&expected),
        "the message names the expected digest: {}",
        refused.message
    );
    assert!(
        refused.message.contains(&hello),
        "the message names the digest the remote file has: {}",
        refused.message
    );

    let uploads = jiuwenbox_requests(&server, "/upload").await;
    assert_eq!(uploads.len(), 1, "only the matching write uploaded");
    assert!(has_query(
        &uploads[0],
        "sandbox_path",
        "/workspace/match.txt"
    ));
    assert!(
        uploads.iter().all(|request| !has_query(
            request,
            "sandbox_path",
            "/workspace/mismatch.txt"
        )),
        "a mismatching digest must never upload"
    );
}

// ─── yuanrong: ADX `file.read` / `file.write` actions ────────────────────────
/// The `file.write` invocations the server recorded.
async fn yuanrong_writes(server: &MockServer) -> Vec<Request> {
    requests(server)
        .await
        .into_iter()
        .filter(|request| body_json(request)["action"] == json!("file.write"))
        .collect()
}

/// `create_if_absent` over the ADX actions: an existing remote path is refused after
/// `file.read` and no `file.write` is issued; a missing one writes.
#[tokio::test]
async fn yuanrong_create_if_absent_refuses_an_existing_file_before_writing() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/direct/env-conditional/invoke"))
        .and(body_partial_json(json!({
            "action": "file.read",
            "args": {"path": "/workspace/existing.txt"},
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": hex("hello")})))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/direct/env-conditional/invoke"))
        .and(body_partial_json(json!({
            "action": "file.read",
            "args": {"path": "/workspace/new.txt"},
        })))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"error": "no such file or directory"})),
        )
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/direct/env-conditional/invoke"))
        .and(body_partial_json(json!({"action": "file.write"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"result": "ok"})))
        .mount(&server)
        .await;

    let isolation = "session_yuanrong_create_if_absent";
    let config = gateway_config(&server, json!({}));
    let fs = fs_operation(
        &registry(),
        "yuanrong",
        &endpoint(&server, Some("env-conditional"), isolation),
        &config,
    );

    let refused = tokio::time::timeout(
        BOUND,
        fs.write_file(create_if_absent("/workspace/existing.txt", "replacement")),
    )
    .await
    .unwrap();
    assert_eq!(refused.code, 199_003);
    assert!(
        refused.message.contains("/workspace/existing.txt"),
        "{}",
        refused.message
    );
    assert!(
        yuanrong_writes(&server).await.is_empty(),
        "a refused precondition issues no file.write action"
    );

    let created = tokio::time::timeout(
        BOUND,
        fs.write_file(create_if_absent("/workspace/new.txt", "hello")),
    )
    .await
    .unwrap();
    assert_eq!(created.code, 0, "{}", created.message);
    let writes = yuanrong_writes(&server).await;
    assert_eq!(writes.len(), 1, "the matching case writes once");
    assert_eq!(
        body_json(&writes[0]),
        json!({
            "action": "file.write",
            "args": {
                "path": "/workspace/new.txt",
                "data": hex("hello"),
                "binary": true,
                "append": false,
            },
        })
    );
}

// ─── aio: kernel gateway envelope ───────────────────────────────────────────
/// `create_if_absent` through the gateway envelope: the read envelope is posted, the
/// refusal happens host-side, and no `write_file` envelope is posted.
#[tokio::test]
async fn aio_create_if_absent_refuses_an_existing_file_without_a_write_envelope() {
    let server = MockServer::start().await;
    mount_aio_data(
        &server,
        json!({
            "path": "/workspace/existing.txt",
            "content": hex("hello"),
            "content_encoding": "hex",
        }),
    )
    .await;

    let isolation = "session_aio_create_if_absent";
    let config = gateway_config(&server, json!({}));
    let fs = fs_operation(
        &registry(),
        "aio",
        &endpoint(&server, None, isolation),
        &config,
    );

    let refused = tokio::time::timeout(
        BOUND,
        fs.write_file(create_if_absent("/workspace/existing.txt", "replacement")),
    )
    .await
    .unwrap();
    assert_eq!(refused.code, 199_003);
    assert!(
        refused.message.contains("/workspace/existing.txt"),
        "{}",
        refused.message
    );
    assert!(
        refused.message.contains("create_if_absent"),
        "{}",
        refused.message
    );

    let recorded = requests(&server).await;
    assert_eq!(
        recorded.len(),
        1,
        "only the read envelope may be posted: {:?}",
        recorded
            .iter()
            .map(|request| request.url.path().to_string())
            .collect::<Vec<_>>()
    );
    let body = body_json(&recorded[0]);
    assert_eq!(body["op_type"], "fs");
    assert_eq!(body["method"], "read_file");
    assert_eq!(body["params"]["path"], "/workspace/existing.txt");
    assert_eq!(body["params"]["mode"], "bytes");
    assert!(
        recorded
            .iter()
            .all(|request| body_json(request)["method"] != json!("write_file")),
        "the refused write must not post a write_file envelope"
    );
}
