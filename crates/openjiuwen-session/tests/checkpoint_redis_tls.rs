//! TLS (`rediss://`) behavior for the Redis checkpoint backend.
//!
//! The clear-text backend contract lives in `checkpoint_redis.rs`; this file
//! covers the TLS transport: the scheme decision, a handshake against a
//! non-TLS server, the full backend contract over a TLS connection, trust
//! anchors (the fixture CA, an unrelated CA, and bytes that are not a
//! certificate), malformed and unreadable CA files, and the declarative
//! factory keys.
//!
//! Everything runs offline against the in-process TLS server in
//! `./support/resp_server.rs`, which serves the static DER fixtures under
//! `tests/fixtures/tls/` (see the README there for their provenance). No
//! assertion depends on wall-clock timing.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_session::checkpoint::redis::DEFAULT_REDIS_KEY_PREFIX;
use openjiuwen_session::checkpoint::resp::RespConnection;
use openjiuwen_session::checkpoint::{
    create_checkpointer, Checkpoint, CheckpointMeta, CheckpointNamespace, Checkpointer,
    CheckpointerConfig, CheckpointerKind,
};
use openjiuwen_session::{RedisCheckpointer, RedisOptions};
use serde_json::json;
use support::resp_server::{fixture, FakeRedis};
use support::{run_backend_contract, ListOrder};

/// The fixture CA path in the form [`RedisOptions::tls_ca_file`] takes.
fn ca_pem() -> String {
    fixture("ca.pem").display().to_string()
}

/// Connects to `url` with `ca_file` as the only extra trust anchor.
async fn connect_with_ca(
    url: &str,
    ca_file: &std::path::Path,
) -> openjiuwen_common::error::Result<RedisCheckpointer> {
    RedisCheckpointer::connect_with_options(RedisOptions {
        url: url.to_string(),
        tls_ca_file: Some(ca_file.display().to_string()),
        ..RedisOptions::default()
    })
    .await
}

#[test]
fn rediss_urls_select_tls_and_redis_urls_do_not() {
    assert!(RespConnection::uses_tls("rediss://localhost:6379").unwrap());
    assert!(!RespConnection::uses_tls("redis://localhost:6379").unwrap());
    // The scheme is the only difference: credentials, ports, bracketed IPv6
    // hosts and database paths keep parsing exactly the same way (the parse
    // details are unit-tested in `src/checkpoint/resp.rs`).
    assert!(RespConnection::uses_tls("rediss://user:secret@[::1]:6390/2").unwrap());
    assert!(!RespConnection::uses_tls("redis://user:secret@[::1]:6390/2").unwrap());
    // A malformed TLS URL is still a validation error, not a connection try.
    for malformed in [
        "rediss://",
        "rediss://:6379",
        "rediss://host:not-a-port",
        "rediss://host/not-a-database",
        "http://host:6379",
    ] {
        assert_eq!(
            RespConnection::uses_tls(malformed).unwrap_err().kind(),
            ErrorKind::Validation,
            "unexpected error for {malformed:?}"
        );
    }
}

#[tokio::test]
async fn tls_against_a_non_tls_server_is_a_session_error() {
    let server = FakeRedis::start().await;
    let error = RedisCheckpointer::connect(server.url_tls())
        .await
        .expect_err("a TLS client must not fall back to clear text");
    assert_eq!(error.kind(), ErrorKind::Session);
    assert_ne!(error.kind(), ErrorKind::Unsupported);
    let message = error.to_string();
    assert!(message.contains("TLS handshake"), "{message}");
    assert!(
        message.contains("127.0.0.1"),
        "the host is named: {message}"
    );
    // The handshake failed before any command was framed as RESP2.
    assert_eq!(server.count("GET"), 0);
    assert_eq!(server.count("PING"), 0);
}

#[tokio::test]
async fn tls_backend_satisfies_the_contract() {
    let server = FakeRedis::start_tls().await;
    let checkpointer = connect_with_ca(&server.url_tls(), &fixture("ca.pem"))
        .await
        .expect("the fixture CA must trust the fixture server");
    assert_eq!(checkpointer.options().url, server.url_tls());
    assert_eq!(checkpointer.options().prefix, DEFAULT_REDIS_KEY_PREFIX);
    assert_eq!(
        checkpointer.options().tls_ca_file.as_deref(),
        Some(ca_pem().as_str())
    );
    // The same contract the clear-text backend passes, over TLS.
    run_backend_contract("redis-tls", Arc::new(checkpointer), ListOrder::OldestFirst).await;
}

#[tokio::test]
async fn tls_save_load_list_and_delete_round_trip() {
    let server = FakeRedis::start_tls().await;
    let checkpointer = RedisCheckpointer::connect_with_options(RedisOptions {
        url: server.url_tls(),
        prefix: "tls".to_string(),
        ttl: Some(Duration::from_secs(60)),
        tls_ca_file: Some(ca_pem()),
        ..RedisOptions::default()
    })
    .await
    .unwrap();
    let checkpoint_id = checkpointer
        .save(
            Checkpoint::new("session-tls", CheckpointNamespace::Agent, "agent-1")
                .with_global_state(json!({ "step": 1 })),
        )
        .await
        .unwrap();

    let loaded = checkpointer
        .load("session-tls", &checkpoint_id)
        .await
        .unwrap()
        .expect("the saved checkpoint loads back");
    assert_eq!(loaded.session_id, "session-tls");
    assert_eq!(loaded.entity_id, "agent-1");
    assert_eq!(loaded.global_state, json!({ "step": 1 }));

    let listed = checkpointer.list("session-tls").await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].checkpoint_id, checkpoint_id);
    assert!(checkpointer.session_exists("session-tls").await.unwrap());

    // The key layout and the TTL are the same as over clear text.
    let set = server
        .command("SET")
        .expect("the backend writes the checkpoint document");
    assert_eq!(
        set[1],
        format!("tls:session:session-tls:checkpoint:{checkpoint_id}")
    );
    let expire = server
        .command("EXPIRE")
        .expect("a configured TTL sends EXPIRE");
    assert_eq!(
        expire[1],
        format!("tls:session:session-tls:checkpoint:{checkpoint_id}")
    );
    assert_eq!(expire[2], "60");

    checkpointer
        .delete("session-tls", &checkpoint_id)
        .await
        .unwrap();
    assert!(checkpointer
        .load("session-tls", &checkpoint_id)
        .await
        .unwrap()
        .is_none());
    assert_eq!(
        checkpointer.list("session-tls").await.unwrap(),
        Vec::<CheckpointMeta>::new()
    );
    assert!(!checkpointer.session_exists("session-tls").await.unwrap());

    // Releasing an unknown session stays a no-op, exactly like clear text.
    checkpointer.release("session-tls").await.unwrap();
}

#[tokio::test]
async fn an_untrusted_certificate_is_a_session_error() {
    let server = FakeRedis::start_tls().await;
    // A second, perfectly valid CA that did not sign this server's certificate.
    for ca_file in [fixture("other-ca.pem"), fixture("other-ca.der")] {
        let error = connect_with_ca(&server.url_tls(), &ca_file)
            .await
            .expect_err("an unrelated CA must not be trusted");
        assert_eq!(error.kind(), ErrorKind::Session, "{ca_file:?}");
        assert!(
            error.to_string().contains("127.0.0.1"),
            "the host is named for {ca_file:?}: {error}"
        );
    }

    // Bytes that are not a certificate at all: the trust store cannot be built,
    // which is still a Session error (the connections cannot be trusted).
    let directory = tempfile::tempdir().unwrap();
    let bogus = directory.path().join("bogus.der");
    std::fs::write(&bogus, b"this is not a certificate").unwrap();
    let error = connect_with_ca(&server.url_tls(), &bogus)
        .await
        .expect_err("a bogus DER file must not be trusted");
    assert_eq!(error.kind(), ErrorKind::Session);
    assert!(error.to_string().contains("127.0.0.1"), "{error}");
}

#[tokio::test]
async fn a_malformed_ca_file_is_a_validation_error() {
    let server = FakeRedis::start_tls().await;
    let directory = tempfile::tempdir().unwrap();

    let malformed = directory.path().join("malformed.pem");
    std::fs::write(
        &malformed,
        "-----BEGIN CERTIFICATE-----\nnot base64!\n-----END CERTIFICATE-----\n",
    )
    .unwrap();
    let error = connect_with_ca(&server.url_tls(), &malformed)
        .await
        .expect_err("a broken PEM bundle must not be trusted");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("base64"), "{error}");

    let empty = directory.path().join("empty.pem");
    std::fs::write(&empty, "").unwrap();
    let error = connect_with_ca(&server.url_tls(), &empty)
        .await
        .expect_err("an empty CA file must not be trusted");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("empty"), "{error}");

    // An unreadable file is a transport-level Session error, not a validation one.
    let missing = directory.path().join("missing.pem");
    let error = connect_with_ca(&server.url_tls(), &missing)
        .await
        .expect_err("a missing CA file must not be trusted");
    assert_eq!(error.kind(), ErrorKind::Session);
    assert!(error.to_string().contains("failed to read"), "{error}");
}

#[tokio::test]
async fn the_factory_configuration_carries_the_ca_file() {
    let server = FakeRedis::start_tls().await;
    let config = CheckpointerConfig {
        kind: CheckpointerKind::Redis,
        path: None,
        extra: BTreeMap::from([
            ("url".to_string(), json!(server.url_tls())),
            ("prefix".to_string(), json!("factory-tls")),
            ("tls_ca_file".to_string(), json!(ca_pem())),
        ]),
    };
    let options = RedisOptions::from_config(&config).expect("the extra map names the CA file");
    assert_eq!(
        options.url,
        server.url_tls(),
        "the factory keeps the rediss url"
    );
    assert_eq!(options.prefix, "factory-tls");
    assert_eq!(options.tls_ca_file.as_deref(), Some(ca_pem().as_str()));

    let checkpointer = RedisCheckpointer::connect_with_options(options)
        .await
        .expect("the extra CA file trusts the fixture server");
    let checkpoint_id = checkpointer
        .save(Checkpoint::new(
            "session-factory-tls",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap();
    assert!(checkpointer
        .load("session-factory-tls", &checkpoint_id)
        .await
        .unwrap()
        .is_some());

    // The hyphenated spelling is an alias for the same key.
    let mut hyphens = config.clone();
    hyphens.extra.remove("tls_ca_file");
    hyphens
        .extra
        .insert("tls-ca-file".to_string(), json!(ca_pem()));
    assert_eq!(
        RedisOptions::from_config(&hyphens).unwrap().tls_ca_file,
        Some(ca_pem())
    );

    // Without the key the factory falls back to the compiled-in roots, which do
    // not include the self-signed fixture CA: a Session error, never Unsupported.
    let without_ca = CheckpointerConfig {
        kind: CheckpointerKind::Redis,
        path: None,
        extra: BTreeMap::from([("url".to_string(), json!(server.url_tls()))]),
    };
    let error = create_checkpointer(&without_ca)
        .await
        .expect_err("an untrusted certificate must fail the factory");
    assert_eq!(error.kind(), ErrorKind::Session);
    assert!(error.to_string().contains("127.0.0.1"), "{error}");
}

#[tokio::test]
async fn a_ca_file_without_a_tls_url_is_rejected() {
    let server = FakeRedis::start().await;
    let error = connect_with_ca(&server.url(), &fixture("ca.pem"))
        .await
        .expect_err("a clear-text connection has no TLS layer to verify");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("tls_ca_file"), "{error}");
}
