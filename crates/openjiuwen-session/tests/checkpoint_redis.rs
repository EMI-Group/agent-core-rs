//! Behavior tests for the Redis checkpoint backend.
//!
//! Ports the storage contract of `openjiuwen/core/session/checkpointer/`
//! (v0.1.20) onto the Rust RESP2 client: the shared backend contract lives in
//! `./support/mod.rs`, and this file adds the Redis-specific behavior (key
//! layout, TTL, prefix isolation, cross-session resume, factory wiring, URL
//! validation). Everything runs against the in-process fake server in
//! `./support/resp_server.rs`, so the suite is offline and deterministic.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use openjiuwen_common::error::ErrorKind;
use openjiuwen_session::checkpoint::redis::DEFAULT_REDIS_KEY_PREFIX;
use openjiuwen_session::checkpoint::resp::RespConnection;
use openjiuwen_session::checkpoint::{
    create_checkpointer, Checkpoint, CheckpointMeta, CheckpointNamespace, Checkpointer,
    CheckpointerConfig, CheckpointerKind,
};
use openjiuwen_session::{RedisCheckpointer, RedisOptions};
use serde_json::json;
use support::resp_server::FakeRedis;
use support::{run_backend_contract, ListOrder};

/// Saves a checkpoint with an explicit id and creation time.
async fn save_at(
    checkpointer: &dyn Checkpointer,
    session_id: &str,
    checkpoint_id: &str,
    created_at: &str,
) -> String {
    let mut checkpoint = Checkpoint::new(session_id, CheckpointNamespace::Workflow, "flow-1")
        .with_checkpoint_id(checkpoint_id)
        .with_global_state(json!({ "counter": 1 }));
    checkpoint.created_at = created_at.to_string();
    checkpointer.save(checkpoint).await.unwrap()
}

#[tokio::test]
async fn redis_backend_satisfies_the_contract() {
    let server = FakeRedis::start().await;
    let checkpointer = RedisCheckpointer::connect(server.url()).await.unwrap();
    assert_eq!(checkpointer.options().prefix, DEFAULT_REDIS_KEY_PREFIX);
    run_backend_contract("redis", Arc::new(checkpointer), ListOrder::OldestFirst).await;
}

#[tokio::test]
async fn saves_documents_under_the_documented_keys() {
    let server = FakeRedis::start().await;
    let checkpointer = RedisCheckpointer::connect(server.url()).await.unwrap();
    let checkpoint_id = save_at(
        &checkpointer,
        "session-keys",
        "cp-keys",
        "2024-01-01T00:00:00.000Z",
    )
    .await;
    assert_eq!(checkpoint_id, "cp-keys");

    let document = format!("{DEFAULT_REDIS_KEY_PREFIX}:session:session-keys:checkpoint:cp-keys");
    let set = server
        .command("SET")
        .expect("the backend sets the checkpoint document");
    assert_eq!(set[0], "SET");
    assert_eq!(set[1], document);
    let payload: serde_json::Value = serde_json::from_str(&set[2]).unwrap();
    assert_eq!(payload["session_id"], "session-keys");
    assert_eq!(payload["checkpoint_id"], "cp-keys");
    assert_eq!(payload["global_state"]["counter"], 1);

    let sadd: Vec<Vec<String>> = server
        .commands()
        .into_iter()
        .filter(|command| command.first().is_some_and(|name| name == "SADD"))
        .collect();
    assert_eq!(sadd.len(), 2, "one index write per save: {sadd:?}");
    assert_eq!(
        sadd[0],
        vec![
            "SADD".to_string(),
            format!("{DEFAULT_REDIS_KEY_PREFIX}:session:session-keys:checkpoints"),
            "cp-keys".to_string(),
        ]
    );
    assert_eq!(
        sadd[1],
        vec![
            "SADD".to_string(),
            format!("{DEFAULT_REDIS_KEY_PREFIX}:sessions"),
            "session-keys".to_string(),
        ]
    );
    // No TTL configured, so no EXPIRE is sent.
    assert_eq!(server.count("EXPIRE"), 0);
}

#[tokio::test]
async fn ttl_option_expires_each_checkpoint() {
    let server = FakeRedis::start().await;
    let checkpointer = RedisCheckpointer::connect_with_options(RedisOptions {
        url: server.url(),
        ttl: Some(Duration::from_secs(90)),
        ..RedisOptions::default()
    })
    .await
    .unwrap();
    let checkpoint_id = checkpointer
        .save(Checkpoint::new(
            "session-ttl",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap();
    let expire = server
        .command("EXPIRE")
        .expect("a configured TTL sends EXPIRE");
    assert_eq!(
        expire,
        vec![
            "EXPIRE".to_string(),
            format!("{DEFAULT_REDIS_KEY_PREFIX}:session:session-ttl:checkpoint:{checkpoint_id}"),
            "90".to_string(),
        ]
    );
    // A sub-second TTL is rounded up so the document is never deleted outright.
    let rounds_up = RedisCheckpointer::connect_with_options(RedisOptions {
        url: server.url(),
        prefix: "round-up".to_string(),
        ttl: Some(Duration::from_millis(10)),
        ..RedisOptions::default()
    })
    .await
    .unwrap();
    rounds_up
        .save(Checkpoint::new(
            "session-ttl",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap();
    let last = server
        .commands()
        .into_iter()
        .rfind(|command| command.first().is_some_and(|name| name == "EXPIRE"))
        .expect("the second backend also expires its documents");
    assert_eq!(last[2], "1");
}

#[tokio::test]
async fn shared_checkpointer_resumes_across_sessions() {
    let server = FakeRedis::start().await;
    let writer = Arc::new(RedisCheckpointer::connect(server.url()).await.unwrap());
    let reader = Arc::new(RedisCheckpointer::connect(server.url()).await.unwrap());

    let mut checkpoint = Checkpoint::new("session-a", CheckpointNamespace::Workflow, "flow-1")
        .with_checkpoint_id("cp-resume")
        .with_global_state(json!({ "step": 2 }))
        .with_component_state(json!({ "node_a": { "value": "kept" } }));
    checkpoint.created_at = "2024-05-01T00:00:00.000Z".to_string();
    writer.save(checkpoint).await.unwrap();

    // A second client (a resumed process) sees the same document.
    let loaded = reader
        .load("session-a", "cp-resume")
        .await
        .unwrap()
        .expect("the writer's checkpoint is readable by the second client");
    assert_eq!(loaded.session_id, "session-a");
    assert_eq!(loaded.global_state, json!({ "step": 2 }));
    assert_eq!(
        loaded.component_state,
        json!({ "node_a": { "value": "kept" } })
    );
    let latest = reader
        .latest("session-a")
        .await
        .unwrap()
        .expect("the session has a latest checkpoint");
    assert_eq!(latest.checkpoint_id, "cp-resume");
    assert!(reader.session_exists("session-a").await.unwrap());
    assert!(!reader.session_exists("session-b").await.unwrap());

    // Releasing on one client clears the session for both.
    reader.release("session-a").await.unwrap();
    assert!(!writer.session_exists("session-a").await.unwrap());
    assert!(writer
        .load("session-a", "cp-resume")
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn prefix_isolates_checkpointers_on_one_server() {
    let server = FakeRedis::start().await;
    let alpha = RedisCheckpointer::connect_with_options(RedisOptions {
        url: server.url(),
        prefix: "alpha".to_string(),
        ..RedisOptions::default()
    })
    .await
    .unwrap();
    let beta = RedisCheckpointer::connect_with_options(RedisOptions {
        url: server.url(),
        prefix: "beta".to_string(),
        ..RedisOptions::default()
    })
    .await
    .unwrap();

    let checkpoint_id = alpha
        .save(Checkpoint::new(
            "shared-session",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap();
    assert!(alpha.session_exists("shared-session").await.unwrap());
    assert_eq!(alpha.list("shared-session").await.unwrap().len(), 1);

    assert!(!beta.session_exists("shared-session").await.unwrap());
    assert_eq!(
        beta.list("shared-session").await.unwrap(),
        Vec::<CheckpointMeta>::new()
    );
    assert!(beta
        .load("shared-session", &checkpoint_id)
        .await
        .unwrap()
        .is_none());
    assert!(beta.latest("shared-session").await.unwrap().is_none());
    let keys: Vec<String> = server
        .commands()
        .into_iter()
        .filter_map(|command| command.get(1).cloned())
        .collect();
    assert!(
        keys.iter()
            .any(|key| key.starts_with("alpha:session:shared-session")),
        "{keys:?}"
    );
}

#[tokio::test]
async fn factory_builds_a_redis_checkpointer_from_extra() {
    let server = FakeRedis::start().await;
    let config = CheckpointerConfig {
        kind: CheckpointerKind::Redis,
        path: None,
        extra: BTreeMap::from([
            ("url".to_string(), json!(server.url())),
            ("prefix".to_string(), json!("factory")),
            ("ttl_seconds".to_string(), json!(120)),
            ("database".to_string(), json!(3)),
        ]),
    };
    let checkpointer = create_checkpointer(&config).await.unwrap();
    let checkpoint_id = checkpointer
        .save(Checkpoint::new(
            "session-factory",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap();
    assert!(checkpointer
        .load("session-factory", &checkpoint_id)
        .await
        .unwrap()
        .is_some());
    let select = server
        .command("SELECT")
        .expect("the factory selects the database");
    assert_eq!(select, vec!["SELECT".to_string(), "3".to_string()]);
    let set = server
        .command("SET")
        .expect("the backend writes a document");
    assert_eq!(
        set[1],
        format!("factory:session:session-factory:checkpoint:{checkpoint_id}")
    );
    let expire = server
        .command("EXPIRE")
        .expect("ttl_seconds configures EXPIRE");
    assert_eq!(expire[2], "120");

    // `CheckpointerConfig::path` is accepted as the URL when `extra.url` is
    // absent (the Python factory passes a connection URL the same way).
    let from_path = CheckpointerConfig {
        kind: CheckpointerKind::Redis,
        path: Some(server.url().into()),
        extra: BTreeMap::new(),
    };
    let checkpointer = create_checkpointer(&from_path).await.unwrap();
    assert!(checkpointer
        .save(Checkpoint::new(
            "session-from-path",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .is_ok());
}

#[tokio::test]
async fn factory_rejects_a_missing_url() {
    let error = create_checkpointer(&CheckpointerConfig {
        kind: CheckpointerKind::Redis,
        path: None,
        extra: BTreeMap::new(),
    })
    .await
    .expect_err("a redis checkpointer without a url must fail");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("url"), "{error}");
}

#[tokio::test]
async fn connect_rejects_malformed_urls() {
    for malformed in [
        String::new(),
        "http://127.0.0.1:6379".to_string(),
        "127.0.0.1:6379".to_string(),
        "redis://127.0.0.1:not-a-port".to_string(),
        "redis://127.0.0.1:6379/not-a-database".to_string(),
        "rediss://127.0.0.1:not-a-port".to_string(),
        "rediss://127.0.0.1:6379/not-a-database".to_string(),
        "rediss://".to_string(),
    ] {
        let error = RedisCheckpointer::connect(malformed.clone())
            .await
            .err()
            .unwrap_or_else(|| panic!("the url {malformed:?} must be rejected"));
        assert_eq!(
            error.kind(),
            ErrorKind::Validation,
            "unexpected error for {malformed:?}: {error}"
        );
    }
    // `rediss://` became a real TLS transport, so it is no longer refused as an
    // unsupported feature; see `checkpoint_redis_tls.rs` for the TLS coverage.
    assert!(!RespConnection::uses_tls("redis://127.0.0.1:6379").unwrap());
    assert!(RespConnection::uses_tls("rediss://127.0.0.1:6379").unwrap());
}

#[tokio::test]
async fn authenticates_and_selects_the_database_from_the_url() {
    let server = FakeRedis::start().await;
    let checkpointer = RedisCheckpointer::connect(format!("{}/2", server.url_with("user:secret")))
        .await
        .unwrap();
    let auth = server
        .command("AUTH")
        .expect("credentials in the URL send AUTH");
    assert_eq!(
        auth,
        vec!["AUTH".to_string(), "user".to_string(), "secret".to_string()]
    );
    let select = server
        .command("SELECT")
        .expect("a database in the URL sends SELECT");
    assert_eq!(select, vec!["SELECT".to_string(), "2".to_string()]);
    assert_eq!(checkpointer.options().database, None);
    assert!(!checkpointer.session_exists("session-auth").await.unwrap());
}

#[tokio::test]
async fn reconnects_after_the_server_drops_the_connection() {
    let server = FakeRedis::start().await;
    let checkpointer = RedisCheckpointer::connect(server.url()).await.unwrap();
    checkpointer
        .save(Checkpoint::new(
            "session-retry",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap();
    // Every connection is closed right after a reply from now on, so the cached
    // connection is dead and the backend must dial again and retry.
    server.hang_up_after_each_reply();
    let checkpoint_id = checkpointer
        .save(Checkpoint::new(
            "session-retry",
            CheckpointNamespace::Agent,
            "agent-2",
        ))
        .await
        .expect("the backend reconnects once after a transport failure");
    let loaded = checkpointer
        .load("session-retry", &checkpoint_id)
        .await
        .expect("reads reconnect as well")
        .expect("the retried save stored the checkpoint");
    assert_eq!(loaded.entity_id, "agent-2");
    assert_eq!(checkpointer.list("session-retry").await.unwrap().len(), 2);
}

#[tokio::test]
async fn connection_failures_are_session_errors() {
    // Bind and release a loopback port so the next connect is refused.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a throwaway listener");
    let addr = listener.local_addr().expect("read the throwaway address");
    drop(listener);
    let error = RedisCheckpointer::connect(format!("redis://{addr}"))
        .await
        .expect_err("connecting to a closed port must fail");
    assert_eq!(error.kind(), ErrorKind::Session);
    assert!(error.to_string().contains("failed to connect"), "{error}");
}

#[tokio::test]
async fn prunes_expired_documents_from_the_index() {
    let server = FakeRedis::start().await;
    let checkpointer = RedisCheckpointer::connect(server.url()).await.unwrap();
    save_at(
        &checkpointer,
        "session-stale",
        "cp-stale",
        "2024-01-01T00:00:00.000Z",
    )
    .await;
    // Delete the document behind the index's back, as an expiring TTL would.
    let document = format!("{DEFAULT_REDIS_KEY_PREFIX}:session:session-stale:checkpoint:cp-stale");
    let mut raw = RespConnection::connect(&server.url()).await.unwrap();
    raw.command(&["DEL", &document]).await.unwrap();
    assert_eq!(
        checkpointer.list("session-stale").await.unwrap(),
        Vec::<CheckpointMeta>::new()
    );
    assert!(
        server
            .commands()
            .iter()
            .any(|command| command.first().is_some_and(|name| name == "SREM")),
        "the stale index member is pruned"
    );
}

/// The deadline the timeout tests configure: long enough that a loaded machine
/// does not trip it early, short enough to keep the suite fast.
const IO_BUDGET: Duration = Duration::from_millis(250);

/// The bound each awaited call gets: a call that hangs fails the test instead of
/// hanging the suite (the runner would otherwise wait forever).
const BOUND: Duration = Duration::from_secs(5);

/// A TCP listener that accepts connections and never answers, standing in for a
/// partitioned or blackholed Redis server.
///
/// Accepted sockets are kept open and left silent: the client's write fits in
/// the socket buffer, so it then waits on a reply that never comes.
struct Blackhole {
    addr: String,
    accepts: Arc<AtomicUsize>,
    handle: tokio::task::JoinHandle<()>,
}

impl Drop for Blackhole {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

impl Blackhole {
    async fn start() -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind the blackhole listener");
        let addr = listener
            .local_addr()
            .expect("read the blackhole address")
            .to_string();
        let accepts = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&accepts);
        let handle = tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((stream, _)) = listener.accept().await {
                counter.fetch_add(1, Ordering::SeqCst);
                held.push(stream);
            }
        });
        Self {
            addr,
            accepts,
            handle,
        }
    }

    fn url(&self) -> String {
        format!("redis://{}", self.addr)
    }

    /// How many client connections were accepted so far.
    fn accepts(&self) -> usize {
        self.accepts.load(Ordering::SeqCst)
    }
}

/// Saves one checkpoint, failing the test when the call outlives [`BOUND`].
async fn bounded_save(
    checkpointer: &RedisCheckpointer,
    session_id: &str,
) -> openjiuwen_common::error::Error {
    let call = checkpointer.save(Checkpoint::new(
        session_id,
        CheckpointNamespace::Agent,
        "agent-hung",
    ));
    tokio::time::timeout(BOUND, call)
        .await
        .unwrap_or_else(|_| panic!("save against a silent server must not hang"))
        .expect_err("a silent server cannot store a checkpoint")
}

/// Waits until the blackhole has accepted `expected` connections.
///
/// The accept counter is bumped by the listener task, so it can lag one
/// scheduling tick behind the client's completed dial.
async fn wait_for_accepts(server: &Blackhole, expected: usize) -> usize {
    let deadline = Instant::now() + BOUND;
    loop {
        let accepted = server.accepts();
        if accepted >= expected {
            return accepted;
        }
        assert!(
            Instant::now() < deadline,
            "expected {expected} accepted connections, saw {accepted}"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Lets a pending accept be recorded before asserting that none happened.
async fn settle() {
    tokio::time::sleep(Duration::from_millis(20)).await;
}

/// Connects a checkpointer with a short, explicit deadline.
async fn timed_checkpointer(server: &Blackhole) -> RedisCheckpointer {
    let checkpointer = RedisCheckpointer::connect_with_options(RedisOptions {
        url: server.url(),
        io_timeout: Some(IO_BUDGET),
        ..RedisOptions::default()
    })
    .await
    .expect("the blackhole accepts the TCP connection, so the eager dial succeeds");
    assert_eq!(checkpointer.options().effective_io_timeout(), IO_BUDGET);
    checkpointer
}

#[tokio::test]
async fn a_silent_server_times_out_instead_of_hanging() {
    let server = Blackhole::start().await;
    let checkpointer = timed_checkpointer(&server).await;

    let started = Instant::now();
    let error = bounded_save(&checkpointer, "session-hung").await;
    assert_eq!(error.kind(), ErrorKind::Timeout, "{error}");
    assert!(error.to_string().contains("timed out"), "{error}");
    assert!(
        started.elapsed() < BOUND,
        "the deadline is the configured one, not the test bound"
    );

    // Reads are bounded the same way.
    let error = tokio::time::timeout(BOUND, checkpointer.load("session-hung", "cp-1"))
        .await
        .expect("load against a silent server must not hang")
        .expect_err("a silent server cannot answer a read");
    assert_eq!(error.kind(), ErrorKind::Timeout, "{error}");

    // ...and so are the index scans behind `list`/`session_exists`.
    let error = tokio::time::timeout(BOUND, checkpointer.list("session-hung"))
        .await
        .expect("list against a silent server must not hang")
        .expect_err("a silent server cannot answer an index scan");
    assert_eq!(error.kind(), ErrorKind::Timeout, "{error}");
}

#[tokio::test]
async fn a_hung_exchange_does_not_park_concurrent_callers() {
    let server = Blackhole::start().await;
    let checkpointer = Arc::new(timed_checkpointer(&server).await);
    assert_eq!(
        wait_for_accepts(&server, 1).await,
        1,
        "the eager connect dials once"
    );

    let started = Instant::now();
    let first = tokio::spawn({
        let checkpointer = Arc::clone(&checkpointer);
        async move { bounded_save(&checkpointer, "session-first").await }
    });
    let second = tokio::spawn({
        let checkpointer = Arc::clone(&checkpointer);
        async move { bounded_save(&checkpointer, "session-second").await }
    });

    // Both callers dialed while the first exchange was still waiting: the
    // connection mutex guards only the pooled handle, so the second caller does
    // not sit behind a dead connection. The window is half the deadline: a
    // serialized caller could only dial after the first one timed out.
    let both_dialed = tokio::time::timeout(IO_BUDGET / 2, async {
        while server.accepts() < 2 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
    assert!(
        both_dialed.is_ok(),
        "the second caller dialed only after the first timed out (accepts = {})",
        server.accepts()
    );

    let first = tokio::time::timeout(BOUND, first)
        .await
        .expect("the first save must not hang")
        .expect("the first save task must finish");
    let second = tokio::time::timeout(BOUND, second)
        .await
        .expect("the second save must not hang")
        .expect("the second save task must finish");
    assert_eq!(first.kind(), ErrorKind::Timeout, "{first}");
    assert_eq!(second.kind(), ErrorKind::Timeout, "{second}");
    assert!(
        started.elapsed() < IO_BUDGET * 2,
        "the two callers time out concurrently; took {:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn a_timed_out_connection_is_never_reused() {
    let server = Blackhole::start().await;
    let checkpointer = timed_checkpointer(&server).await;
    assert_eq!(
        wait_for_accepts(&server, 1).await,
        1,
        "the eager connect dials once"
    );

    // The first call rides the pooled connection and hits its deadline.
    let error = bounded_save(&checkpointer, "session-discard").await;
    assert_eq!(error.kind(), ErrorKind::Timeout, "{error}");
    settle().await;
    assert_eq!(
        server.accepts(),
        1,
        "the pooled connection served the timed-out call"
    );

    // The timed-out socket could still deliver its late reply, so it must be
    // dropped: the next call dials a fresh connection instead of reading that
    // reply as its own answer.
    let error = bounded_save(&checkpointer, "session-discard").await;
    assert_eq!(error.kind(), ErrorKind::Timeout, "{error}");
    assert_eq!(
        wait_for_accepts(&server, 2).await,
        2,
        "a timed-out connection is discarded, so the next call dialed"
    );
}
