#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use crate::spawn::protocol::CancelRequest;
use openjiuwen_common::error::ErrorKind;

fn assert_backend<T: SpawnedAgent>() {}
fn assert_spawner<T: AgentSpawner>() {}

#[test]
fn the_backend_types_satisfy_the_seam() {
    assert_backend::<SpawnedProcess>();
    assert_spawner::<SubprocessSpawner>();
}

#[test]
fn config_defaults_match_the_contract() {
    let config = SubprocessConfig::default();
    assert_eq!(config.program, "");
    assert_eq!(config.args, Vec::<String>::new());
    assert_eq!(config.cwd, None);
    assert_eq!(config.env, BTreeMap::new());
    assert!((config.request_timeout_secs - 30.0).abs() < f64::EPSILON);
    assert!((config.shutdown_timeout_secs - 10.0).abs() < f64::EPSILON);
    assert!((config.health_check_interval_secs - 5.0).abs() < f64::EPSILON);
    assert!((config.health_check_timeout_secs - 5.0).abs() < f64::EPSILON);
    assert_eq!(config.max_health_failures, 3);
    assert_eq!(config.max_frame_bytes, DEFAULT_MAX_FRAME_BYTES);
}

#[test]
fn config_round_trips_through_json() {
    let config = SubprocessConfig::new("python")
        .with_args(["-m", "agent"])
        .with_cwd("/tmp/agent")
        .with_env("API_KEY", "secret")
        .with_request_timeout(Duration::from_secs(12))
        .with_shutdown_timeout(Duration::from_secs(4))
        .with_health_check(Duration::from_secs(2), Duration::from_millis(500), 7)
        .with_max_frame_bytes(4096);
    let value = serde_json::to_value(&config).unwrap();
    assert_eq!(value["program"], json!("python"));
    assert_eq!(value["args"], json!(["-m", "agent"]));
    assert_eq!(value["cwd"], json!("/tmp/agent"));
    assert_eq!(value["max_frame_bytes"], json!(4096));
    assert_eq!(
        serde_json::from_value::<SubprocessConfig>(value).unwrap(),
        config
    );
}

#[test]
fn config_tolerates_unknown_fields_and_partial_json() {
    let config: SubprocessConfig = serde_json::from_value(json!({
        "program": "node",
        "unknown_field": {"nested": true},
    }))
    .unwrap();
    assert_eq!(config.program, "node");
    assert_eq!(config.request_timeout_secs, 30.0);
    assert_eq!(config.max_frame_bytes, DEFAULT_MAX_FRAME_BYTES);
    let empty: SubprocessConfig = serde_json::from_value(json!({})).unwrap();
    assert_eq!(empty, SubprocessConfig::default());
}

#[test]
fn builders_chain() {
    let config = SubprocessConfig::new("agent")
        .with_arg("--verbose")
        .with_arg("--json");
    assert_eq!(
        config.args,
        vec!["--verbose".to_owned(), "--json".to_owned()]
    );
    assert_eq!(
        config.with_args(["a", "b"]).args,
        vec!["a".to_owned(), "b".to_owned()]
    );
}

#[test]
fn timeout_seconds_are_clamped() {
    assert_eq!(secs(-1.0), Duration::ZERO);
    assert_eq!(secs(f64::NAN), Duration::ZERO);
    assert_eq!(secs(0.0), Duration::ZERO);
    assert_eq!(secs(1.5), Duration::from_secs_f64(1.5));
    assert_eq!(
        secs(MAX_TIMEOUT_SECS * 4.0),
        Duration::from_secs_f64(MAX_TIMEOUT_SECS)
    );
    assert!(secs(f64::INFINITY) <= Duration::from_secs_f64(MAX_TIMEOUT_SECS));
}

#[test]
fn the_shutdown_deadline_is_a_clamped_accessor_not_dead_config() {
    assert_eq!(
        SubprocessConfig::default().shutdown_timeout(),
        Duration::from_secs(10)
    );
    assert_eq!(
        SubprocessConfig::new("agent")
            .with_shutdown_timeout(Duration::from_millis(250))
            .shutdown_timeout(),
        Duration::from_millis(250),
        "the builder reaches the accessor the supervisor reads"
    );
    let mut config = SubprocessConfig::new("agent");
    config.shutdown_timeout_secs = -1.0;
    assert_eq!(config.shutdown_timeout(), Duration::ZERO);
    config.shutdown_timeout_secs = MAX_TIMEOUT_SECS * 4.0;
    assert_eq!(
        config.shutdown_timeout(),
        Duration::from_secs_f64(MAX_TIMEOUT_SECS)
    );
}

#[test]
fn merged_env_lets_the_spec_win() {
    let config = SubprocessConfig::new("agent")
        .with_env("SHARED", "config")
        .with_env("CONFIG_ONLY", "1");
    let spec = SpawnSpec::default()
        .with_env("SHARED", "spec")
        .with_env("SPEC_ONLY", "2");
    let env = merged_env(&config, &spec);
    assert_eq!(env.get("SHARED").map(String::as_str), Some("spec"));
    assert_eq!(env.get("CONFIG_ONLY").map(String::as_str), Some("1"));
    assert_eq!(env.get("SPEC_ONLY").map(String::as_str), Some("2"));
    assert_eq!(env.len(), 3);
}

#[test]
fn bootstrap_frame_carries_the_spawn_spec() {
    let spec = SpawnSpec::class_agent("pkg.agents", "Helper").with_session_id("s-1");
    let frame = bootstrap_frame(&spec).unwrap();
    assert_eq!(frame.kind, SpawnMessageKind::Input);
    assert_eq!(
        frame.payload["spawn_spec"]["agent_module"],
        json!("pkg.agents")
    );
    assert_eq!(frame.payload["spawn_spec"]["agent_class"], json!("Helper"));
    assert_eq!(frame.payload["spawn_spec"]["session_id"], json!("s-1"));
    let line = frame.to_line().unwrap();
    // `serde_json`'s float parser is not correctly rounded, so a round-tripped
    // timestamp may differ by one ULP: compare the whole frame except the
    // timestamp, then the timestamp with a tolerance (see `protocol.rs`).
    let restored = SpawnMessage::from_line(&line).unwrap();
    assert_eq!(restored.kind, frame.kind);
    assert_eq!(restored.payload, frame.payload);
    assert_eq!(restored.message_id, frame.message_id);
    assert!(
        (restored.timestamp - frame.timestamp).abs() < 1e-6,
        "timestamp drifted: {} vs {}",
        restored.timestamp,
        frame.timestamp
    );
}

#[test]
fn invoke_frame_wraps_the_inputs() {
    let frame = invoke_frame(&json!({"query": "hi"}));
    assert_eq!(frame.kind, SpawnMessageKind::Input);
    assert_eq!(frame.payload, json!({"inputs": {"query": "hi"}}));
}

#[test]
fn error_text_understands_objects_strings_and_null() {
    assert_eq!(error_text(&json!({"message": "boom"})), "boom");
    assert_eq!(error_text(&json!("boom")), "boom");
    assert_eq!(error_text(&Value::Null), "unspecified error");
    assert_eq!(
        error_text(&json!({"code": 3})),
        json!({"code": 3}).to_string()
    );
}

#[test]
fn health_payload_defaults_to_healthy() {
    assert!(health_payload(&Value::Null));
    assert!(health_payload(&json!({"healthy": true})));
    assert!(!health_payload(&json!({"healthy": false})));
    assert!(health_payload(&json!({})));
    assert!(health_payload(&json!(true)));
    assert!(!health_payload(&json!(false)));
}

#[test]
fn line_preview_is_bounded() {
    let long = "x".repeat(STDERR_PREVIEW_CHARS + 10);
    assert_eq!(
        line_preview(&long).chars().count(),
        STDERR_PREVIEW_CHARS + 1
    );
    assert_eq!(line_preview("short"), "short");
}

#[tokio::test]
async fn an_empty_program_is_a_config_error() {
    let spawner = SubprocessSpawner::new(SubprocessConfig::default());
    let error = spawner
        .spawn_process(SpawnSpec::class_agent("pkg", "Agent"))
        .await
        .expect_err("an empty program cannot launch");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(
        error.to_string().contains("SubprocessConfig::program"),
        "unexpected message: {error}"
    );

    // The trait entry point reports the same thing (`Arc<dyn SpawnedAgent>`
    // has no `Debug`, so the Ok arm is matched rather than expect_err'd).
    match SubprocessSpawner::new(SubprocessConfig::new("   "))
        .spawn(SpawnSpec::default())
        .await
    {
        Ok(_) => panic!("a blank program cannot launch"),
        Err(error) => assert_eq!(error.kind(), ErrorKind::Config),
    }
}

#[test]
fn the_spawner_exposes_its_config_including_the_frame_cap() {
    let config = SubprocessConfig::new("agent").with_max_frame_bytes(1024);
    let spawner = SubprocessSpawner::new(config.clone());
    assert_eq!(spawner.config(), &config);
    assert_eq!(spawner.config().max_frame_bytes, 1024);
    assert_eq!(
        SubprocessSpawner::default().config().max_frame_bytes,
        frames::DEFAULT_MAX_FRAME_BYTES
    );
}

// --- the trait seam over a real child ---------------------------------------

/// Upper bound on any await here, so a broken exchange fails the test instead of
/// stalling the harness.
const BOUND: Duration = Duration::from_secs(10);

/// The libtest filter that selects [`fake_child`] in this very binary.
fn child_filter() -> String {
    let module = module_path!();
    let relative = module
        .split_once("::")
        .map_or(module, |(_crate_name, rest)| rest);
    format!("{relative}::fake_child")
}

/// A launch configuration that re-executes this test binary as the fake child.
fn child_config() -> SubprocessConfig {
    let program = std::env::current_exe()
        .expect("the test binary path is available")
        .to_string_lossy()
        .into_owned();
    SubprocessConfig::new(program)
        .with_args(vec![
            child_filter(),
            "--exact".to_owned(),
            "--ignored".to_owned(),
            "--test-threads=1".to_owned(),
            "--nocapture".to_owned(),
        ])
        // No background probe races the assertions; explicit probes still work.
        .with_health_check(Duration::ZERO, Duration::from_secs(2), 3)
        .with_request_timeout(BOUND)
}

/// [`child_config`] whose child runs in `mode` (`FAKE_CHILD_MODE`).
fn child_config_in(mode: &str) -> SubprocessConfig {
    child_config().with_env("FAKE_CHILD_MODE", mode)
}

/// Writes one protocol frame as a line and flushes it.
fn emit(out: &mut impl std::io::Write, kind: SpawnMessageKind, payload: Value) {
    let frame = SpawnMessage::new(kind, payload);
    let line = frame.to_line().expect("a frame is serializable");
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
}

/// The hermetic protocol child: this test binary re-executed with the filter of
/// this `#[ignore]`d case.
///
/// It answers an invocation with an `output` frame, a `health_check` with a
/// healthy `health_check_response`, and a `shutdown` with `shutdown_ack`.
/// `FAKE_CHILD_MODE=silent_input` ignores invocations (so a caller deadline must
/// expire), `FAKE_CHILD_MODE=silent_shutdown` ignores a shutdown and never exits
/// (so the parent must enforce its deadline and kill it),
/// `FAKE_CHILD_MODE=big_stderr` writes one 50 MiB newline-free stderr line before
/// serving requests, and
/// `FAKE_CHILD_MODE=cancel_echo` holds an invocation open, records the request id
/// it is holding in the file named by `FAKE_CHILD_MARKER` (the parent's barrier:
/// one ordered stdin pipe means the cancel is written after the invoke frame), and
/// answers a `cancel` frame with a `cancel_ack` plus a terminal `done` whose
/// output text echoes the id and whether it matched the pending request.
#[test]
#[ignore = "re-executed as the hermetic protocol child by the backend tests"]
fn fake_child() {
    use std::io::{BufRead, Write as _};
    let mode = std::env::var("FAKE_CHILD_MODE").unwrap_or_default();
    // The request `cancel_echo` is holding open, if any.
    let mut pending: Option<String> = None;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    // libtest prints `running 1 test` and then `test ... ` without a trailing
    // newline before this body runs; closing that line keeps every protocol frame
    // on a line of its own (the parent skips the noise line).
    let _ = writeln!(out);
    let _ = out.flush();
    if mode == "big_stderr" {
        // One 50 MiB newline-free stderr line: a bounded drain must keep only its
        // cap and never let the child block on a full pipe.
        let chunk = vec![b'x'; 64 * 1024];
        let stderr = std::io::stderr();
        let mut err = stderr.lock();
        for _ in 0..800 {
            err.write_all(&chunk).expect("the parent drains stderr");
        }
        err.write_all(b"\n").expect("the parent drains stderr");
        err.flush().expect("the parent drains stderr");
    }
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let Ok(frame) = SpawnMessage::from_line(&line) else {
            continue;
        };
        match frame.kind {
            SpawnMessageKind::Input => {
                // The bootstrap frame carries no work, and `silent_input` never
                // answers anything.
                if frame.payload.get("spawn_spec").is_some() || mode == "silent_input" {
                    continue;
                }
                if mode == "cancel_echo" {
                    // Hold the request open and let the parent know it arrived.
                    // An invocation frame is `{"inputs": <inputs>}`, so the
                    // caller's correlation id travels inside `inputs`.
                    pending = frame
                        .payload
                        .get("inputs")
                        .and_then(|inputs| inputs.get("request_id"))
                        .and_then(Value::as_str)
                        .map(str::to_owned);
                    if let Ok(path) = std::env::var("FAKE_CHILD_MARKER") {
                        let _ = std::fs::write(path, pending.clone().unwrap_or_default());
                    }
                    continue;
                }
                emit(
                    &mut out,
                    SpawnMessageKind::Output,
                    AgentOutput::answer("pong").to_value(),
                );
            }
            SpawnMessageKind::Cancel => {
                let request = CancelRequest::from_frame(&frame)
                    .expect("the parent writes a well-formed cancel frame");
                let matched = pending.as_deref() == Some(request.request_id.as_str());
                emit(
                    &mut out,
                    SpawnMessageKind::CancelAck,
                    json!({
                        "request_id": request.request_id,
                        "reason": request.reason,
                        "matched": matched,
                    }),
                );
                // `done` is parsed as an `AgentOutput`, so the echo travels in its
                // `output` text (unknown fields would be dropped by the parser).
                emit(
                    &mut out,
                    SpawnMessageKind::Done,
                    json!({
                        "output": format!("cancelled {} matched {}", request.request_id, matched),
                        "result_type": "answer",
                    }),
                );
                pending = None;
            }
            SpawnMessageKind::HealthCheck => emit(
                &mut out,
                SpawnMessageKind::HealthCheckResponse,
                json!({"healthy": true}),
            ),
            SpawnMessageKind::Shutdown => {
                if mode == "silent_shutdown" {
                    // Never acknowledge and never exit: only a kill ends this.
                    loop {
                        std::thread::sleep(Duration::from_millis(50));
                    }
                }
                emit(&mut out, SpawnMessageKind::ShutdownAck, Value::Null);
                break;
            }
            _ => {}
        }
    }
}

/// Waits (bounded) for the fake child to record that it read the invoke frame.
///
/// The marker file is the test's synchronization barrier: the child writes it
/// from its single read loop, so once it exists the invoke frame has been
/// consumed and the next frame on that one ordered stdin pipe is the cancel.
async fn wait_for_marker(path: &std::path::Path) {
    let deadline = tokio::time::Instant::now() + BOUND;
    while !path.exists() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the fake child never read the invoke frame"
        );
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(
        std::fs::read_to_string(path).unwrap(),
        "req-42",
        "the child holds the request the caller named"
    );
}

#[tokio::test]
async fn a_cancel_frame_carries_the_request_id_to_the_child() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("invoke-seen");
    let config = child_config_in("cancel_echo")
        .with_env("FAKE_CHILD_MARKER", marker.to_string_lossy().into_owned());
    let agent: Arc<dyn SpawnedAgent> = tokio::time::timeout(
        BOUND,
        SubprocessSpawner::new(config).spawn(SpawnSpec::class_agent("tests.spawn", "FakeAgent")),
    )
    .await
    .expect("spawn must not hang")
    .expect("the fake child launches");
    let sender = Arc::clone(&agent);
    let request = tokio::spawn(async move {
        sender
            .send(json!({"request_id": "req-42", "work": "endless"}))
            .await
    });
    wait_for_marker(&marker).await;
    // A blank id names nothing: it is refused before a byte is written.
    let blank = agent
        .cancel_task_with_request_id("  ", Some("operator stop"))
        .await
        .expect_err("a blank id names nothing");
    assert_eq!(blank.kind(), ErrorKind::Validation);
    agent
        .cancel_task_with_request_id("req-42", Some("operator stop"))
        .await
        .expect("the child receives the cancel frame");
    let output = tokio::time::timeout(BOUND, request)
        .await
        .expect("the cancel must terminate the request")
        .expect("the send task ran")
        .expect("the cancelled request ends cleanly");
    assert_eq!(
        output.output, "cancelled req-42 matched true",
        "the child echoes the id the caller used, so it correlated the request"
    );
    assert!(
        tokio::time::timeout(BOUND, agent.shutdown(Duration::from_secs(5)))
            .await
            .expect("shutdown must not hang")
            .expect("the fake child acknowledges the shutdown")
    );
    // Nothing is left to receive a frame: the child is gone.
    let after = agent
        .cancel_task("req-42")
        .await
        .expect_err("a stopped child receives nothing");
    assert_eq!(after.kind(), ErrorKind::Transport);
    assert!(after.to_string().contains("cancel"), "{after}");
}

#[tokio::test]
async fn a_child_inherits_the_backend_of_the_spawner_that_launched_it() {
    let spec = || SpawnSpec::class_agent("tests.spawn", "FakeAgent");

    // The plain constructor keeps reporting the subprocess backend.
    let plain: Arc<dyn SpawnedAgent> =
        tokio::time::timeout(BOUND, SubprocessSpawner::new(child_config()).spawn(spec()))
            .await
            .expect("spawn must not hang")
            .expect("the fake child launches");
    assert_eq!(plain.backend(), SpawnBackend::Subprocess);
    assert!(
        tokio::time::timeout(BOUND, plain.shutdown(Duration::from_secs(5)))
            .await
            .expect("shutdown must not hang")
            .expect("the fake child acknowledges the shutdown")
    );

    // A wrapper backend labels its children, through the concrete handle and
    // through the trait object a caller actually holds.
    let spawner = SubprocessSpawner::with_backend(child_config(), SpawnBackend::ExternalCommand);
    assert_eq!(spawner.backend(), SpawnBackend::ExternalCommand);
    let process = tokio::time::timeout(BOUND, spawner.spawn_process(spec()))
        .await
        .expect("spawn must not hang")
        .expect("the fake child launches");
    assert_eq!(process.backend(), SpawnBackend::ExternalCommand);
    assert!(
        format!("{process:?}").contains("ExternalCommand"),
        "the Debug impl carries the label: {process:?}"
    );
    let agent: Arc<dyn SpawnedAgent> = process.into_agent();
    assert_eq!(agent.backend(), SpawnBackend::ExternalCommand);
    assert_eq!(
        agent.health_snapshot().backend,
        SpawnBackend::ExternalCommand
    );
    assert!(
        tokio::time::timeout(BOUND, agent.shutdown(Duration::from_secs(5)))
            .await
            .expect("shutdown must not hang")
            .expect("the fake child acknowledges the shutdown")
    );
}

#[tokio::test]
async fn a_spawned_child_reports_its_backend_and_health_through_the_seam() {
    let spawner = SubprocessSpawner::new(child_config());
    assert_eq!(spawner.backend(), SpawnBackend::Subprocess);
    let agent: Arc<dyn SpawnedAgent> = tokio::time::timeout(
        BOUND,
        spawner.spawn(SpawnSpec::class_agent("tests.spawn", "FakeAgent")),
    )
    .await
    .expect("spawn must not hang")
    .expect("the fake child launches");

    // Everything below goes through `Arc<dyn SpawnedAgent>`, which is the shape a
    // caller actually holds.
    assert_eq!(agent.backend(), SpawnBackend::Subprocess);
    assert!(agent.is_alive());
    assert!(
        agent.is_healthy(),
        "a fresh child is healthy before its first probe"
    );
    let healthy = tokio::time::timeout(BOUND, agent.health())
        .await
        .expect("the health probe must not hang")
        .expect("the fake child answers a probe");
    assert!(healthy);

    let snapshot = agent.health_snapshot();
    assert_eq!(snapshot.backend, SpawnBackend::Subprocess);
    assert!(snapshot.alive);
    assert!(snapshot.healthy);
    assert_eq!(snapshot.pid, agent.pid());
    assert!(snapshot.pid.is_some(), "a real child reports its pid");
    assert_eq!(snapshot.last_error, None);

    let output = tokio::time::timeout(BOUND, agent.send(json!({"query": "ping"})))
        .await
        .expect("send must not hang")
        .expect("the fake child answers");
    assert_eq!(output.output, "pong");

    assert!(
        tokio::time::timeout(BOUND, agent.shutdown(Duration::from_secs(5)))
            .await
            .expect("shutdown must not hang")
            .expect("the fake child acknowledges the shutdown")
    );
}

#[tokio::test]
async fn a_single_huge_stderr_line_is_bounded_and_never_blocks_the_child() {
    // The child writes a 50 MiB newline-free stderr line before it serves the
    // invocation. The bounded drain keeps only `STDERR_LINE_CAP_BYTES`, discards
    // the rest, and keeps reading, so the child is never blocked on a full pipe
    // and the exchange completes normally.
    let agent: Arc<dyn SpawnedAgent> = tokio::time::timeout(
        BOUND,
        SubprocessSpawner::new(child_config_in("big_stderr"))
            .spawn(SpawnSpec::class_agent("tests.spawn", "FakeAgent")),
    )
    .await
    .expect("spawn must not hang")
    .expect("the fake child launches");
    let output = tokio::time::timeout(BOUND, agent.send(json!({"query": "ping"})))
        .await
        .expect("the child must not block on its stderr line")
        .expect("the fake child answers after its stderr line");
    assert_eq!(output.output, "pong");
    assert!(
        tokio::time::timeout(BOUND, agent.shutdown(Duration::from_secs(5)))
            .await
            .expect("shutdown must not hang")
            .expect("the fake child acknowledges the shutdown")
    );
}

#[tokio::test]
async fn a_caller_request_deadline_bounds_one_exchange_and_none_uses_the_configuration() {
    // `None` keeps the configured `request_timeout_secs`, so a cooperative child
    // answers exactly as it does for `send`.
    let echo: Arc<dyn SpawnedAgent> = tokio::time::timeout(
        BOUND,
        SubprocessSpawner::new(child_config()).spawn(SpawnSpec::class_agent("tests.spawn", "Fake")),
    )
    .await
    .expect("spawn must not hang")
    .expect("the fake child launches");
    let output = tokio::time::timeout(
        BOUND,
        echo.send_with_timeout(json!({"query": "ping"}), None),
    )
    .await
    .expect("send must not hang")
    .expect("the fake child answers within the configured deadline");
    assert_eq!(output.output, "pong");
    tokio::time::timeout(BOUND, echo.shutdown(Duration::from_secs(5)))
        .await
        .expect("shutdown must not hang")
        .expect("the fake child acknowledges the shutdown");

    // A silent child: the caller deadline (50 ms) expires even though the
    // configured one is 10 s, and the abandoned exchange desynchronizes the
    // handle.
    let silent: Arc<dyn SpawnedAgent> = tokio::time::timeout(
        BOUND,
        SubprocessSpawner::new(child_config_in("silent_input"))
            .spawn(SpawnSpec::class_agent("tests.spawn", "Fake")),
    )
    .await
    .expect("spawn must not hang")
    .expect("the fake child launches");
    let error = tokio::time::timeout(
        BOUND,
        silent.send_with_timeout(json!({"query": "ping"}), Some(Duration::from_millis(50))),
    )
    .await
    .expect("the caller deadline bounds the exchange")
    .expect_err("a silent child never replies");
    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert!(error.to_string().contains("subprocess request"), "{error}");

    let desynchronized = silent
        .send_with_timeout(json!({"query": "ping"}), None)
        .await
        .expect_err("the abandoned exchange desynchronized the handle");
    assert_eq!(desynchronized.kind(), ErrorKind::Validation);
    assert!(
        tokio::time::timeout(BOUND, silent.shutdown(Duration::from_secs(5)))
            .await
            .expect("shutdown must not hang")
            .expect("a desynchronized handle still shuts down")
    );
}

#[tokio::test]
async fn a_zero_shutdown_timeout_uses_the_configured_deadline_and_a_non_zero_one_wins() {
    // A child that never acknowledges a shutdown is killed after the *configured*
    // deadline when the caller passes `Duration::ZERO`.
    let configured = Duration::from_millis(200);
    let silent = SubprocessSpawner::new(
        child_config_in("silent_shutdown").with_shutdown_timeout(configured),
    )
    .spawn_process(SpawnSpec::class_agent("tests.spawn", "Fake"))
    .await
    .expect("the fake child launches");
    let started = std::time::Instant::now();
    let graceful = tokio::time::timeout(BOUND, silent.shutdown(Duration::ZERO))
        .await
        .expect("shutdown must not hang")
        .expect("killing a silent child is not an error");
    let elapsed = started.elapsed();
    assert!(
        !graceful,
        "a child that never acknowledges has to be killed"
    );
    assert!(
        elapsed >= configured,
        "the configured shutdown deadline ({configured:?}) was honoured: {elapsed:?}"
    );

    // A non-zero caller deadline wins over a long configured one.
    let silent =
        SubprocessSpawner::new(child_config_in("silent_shutdown").with_shutdown_timeout(BOUND))
            .spawn_process(SpawnSpec::class_agent("tests.spawn", "Fake"))
            .await
            .expect("the fake child launches");
    let started = std::time::Instant::now();
    let graceful = tokio::time::timeout(BOUND, silent.shutdown(Duration::from_millis(100)))
        .await
        .expect("shutdown must not hang")
        .expect("killing a silent child is not an error");
    assert!(!graceful);
    assert!(
        started.elapsed() < BOUND,
        "the caller deadline won over the configuration: {:?}",
        started.elapsed()
    );
}

#[test]
fn describe_reports_the_launch_settings_and_no_environment_values() {
    let spawner = SubprocessSpawner::new(
        SubprocessConfig::new("/usr/local/bin/agent")
            .with_args(["--serve", "--port", "0"])
            .with_cwd("/tmp/agent")
            .with_env("AGENT_TOKEN", "super-secret-value")
            .with_request_timeout(Duration::from_secs(12))
            .with_shutdown_timeout(Duration::from_secs(4))
            .with_health_check(Duration::from_secs(2), Duration::from_millis(500), 7)
            .with_max_frame_bytes(4096),
    );
    let described: Value = spawner.describe();
    assert_eq!(described["backend"], json!("subprocess"));
    assert_eq!(described["program"], json!("/usr/local/bin/agent"));
    assert_eq!(described["args"], json!(["--serve", "--port", "0"]));
    assert_eq!(described["cwd"], json!("/tmp/agent"));
    assert_eq!(
        described["env_names"],
        json!(["AGENT_TOKEN"]),
        "the environment is reported by name only"
    );
    assert_eq!(described["request_timeout_secs"], json!(12.0));
    assert_eq!(described["shutdown_timeout_secs"], json!(4.0));
    assert_eq!(described["health_check_interval_secs"], json!(2.0));
    assert_eq!(described["health_check_timeout_secs"], json!(0.5));
    assert_eq!(described["max_health_failures"], json!(7));
    assert_eq!(described["max_frame_bytes"], json!(4096));
    let rendered = described.to_string();
    assert!(
        !rendered.contains("super-secret-value"),
        "no environment value leaks: {rendered}"
    );

    // An unset working directory and an empty environment are JSON null/[].
    let plain: Value = SubprocessSpawner::default().describe();
    assert_eq!(plain["backend"], json!("subprocess"));
    assert_eq!(plain["program"], json!(""));
    assert_eq!(plain["args"], json!([]));
    assert_eq!(plain["cwd"], Value::Null);
    assert_eq!(plain["env_names"], json!([]));
}

#[test]
fn a_wrapper_backend_describes_its_own_label() {
    let spawner = SubprocessSpawner::with_backend(
        SubprocessConfig::new("agent"),
        SpawnBackend::ExternalCommand,
    );
    assert_eq!(spawner.describe()["backend"], json!("external_command"));
}

// --- runtime re-tuning of the background health interval --------------------

/// Whether a background health loop is registered for `process`.
async fn health_loop_running(process: &Arc<SpawnedProcess>) -> bool {
    process.state.health_task.lock().await.is_some()
}

/// [`child_config`] whose configured (initial) health interval is `interval`.
fn child_config_with_interval(interval: Duration) -> SubprocessConfig {
    child_config().with_health_check(interval, Duration::from_secs(2), 3)
}

/// A handle over the hermetic child, with `config`.
async fn process_with(config: SubprocessConfig) -> Arc<SpawnedProcess> {
    tokio::time::timeout(
        BOUND,
        SubprocessSpawner::new(config).spawn_process(SpawnSpec::class_agent("tests.spawn", "Fake")),
    )
    .await
    .expect("spawn must not hang")
    .expect("the fake child launches")
}

#[tokio::test]
async fn the_effective_health_interval_starts_from_the_configuration_and_follows_re_tunes() {
    let config = child_config_with_interval(Duration::from_millis(250));
    let spawner = SubprocessSpawner::new(config.clone());
    let process = process_with(config).await;

    assert_eq!(process.health_interval(), Duration::from_millis(250));
    assert!(
        health_loop_running(&process).await,
        "a non-zero configured interval starts the background loop"
    );

    // Zero stops background probing; explicit probes still work.
    tokio::time::timeout(BOUND, process.set_health_interval(Duration::ZERO))
        .await
        .expect("a re-tune must not hang")
        .expect("a live child accepts a re-tune");
    assert_eq!(process.health_interval(), Duration::ZERO);
    assert!(
        !health_loop_running(&process).await,
        "a zero interval leaves no loop running"
    );
    assert!(
        tokio::time::timeout(BOUND, process.health())
            .await
            .expect("a probe must not hang")
            .expect("the fake child answers a probe"),
        "an explicit probe works without a background loop"
    );

    // And back again: the loop uses the value the handle now holds.
    tokio::time::timeout(
        BOUND,
        process.set_health_interval(Duration::from_millis(200)),
    )
    .await
    .expect("a re-tune must not hang")
    .expect("a live child accepts a re-tune");
    assert_eq!(process.health_interval(), Duration::from_millis(200));
    assert!(health_loop_running(&process).await);

    // The configuration is the *initial* value, never rewritten by a re-tune.
    assert!(
        (spawner.config().health_check_interval_secs - 0.25).abs() < f64::EPSILON,
        "a re-tune is a per-handle setting, not a configuration change"
    );

    // The child survived every restart, in sync and still supervised.
    tokio::time::timeout(BOUND, process.set_health_interval(Duration::ZERO))
        .await
        .expect("a re-tune must not hang")
        .expect("a live child accepts a re-tune");
    assert!(process.is_alive(), "a re-tune never restarts the child");
    assert!(process.is_healthy());
    let output = tokio::time::timeout(BOUND, process.send(json!({"query": "after-re-tune"})))
        .await
        .expect("send must not hang")
        .expect("a re-tuned handle is still in sync");
    assert_eq!(output.output, "pong");

    assert!(
        tokio::time::timeout(BOUND, process.shutdown(Duration::from_secs(5)))
            .await
            .expect("shutdown must not hang")
            .expect("the fake child acknowledges the shutdown")
    );
}

#[tokio::test]
async fn repeated_re_tunes_are_idempotent_and_preserve_the_supervision_state() {
    // The configured interval is zero, so only a re-tune starts a loop.
    let process = process_with(child_config()).await;
    assert!(!health_loop_running(&process).await);

    // An unhealthy observation with two consecutive failures, which a re-tune
    // must neither clear nor reset.
    process.state.record_health(false);
    process.state.health_failures.store(2, Ordering::SeqCst);

    for _ in 0..3 {
        tokio::time::timeout(
            BOUND,
            process.set_health_interval(Duration::from_millis(200)),
        )
        .await
        .expect("a re-tune must not hang")
        .expect("a live child accepts a re-tune");
        assert_eq!(process.health_interval(), Duration::from_millis(200));
        assert!(
            health_loop_running(&process).await,
            "each re-tune leaves exactly one loop registered"
        );
    }
    for _ in 0..2 {
        tokio::time::timeout(BOUND, process.set_health_interval(Duration::ZERO))
            .await
            .expect("a re-tune must not hang")
            .expect("a live child accepts a re-tune");
        assert_eq!(process.health_interval(), Duration::ZERO);
        assert!(!health_loop_running(&process).await);
    }

    assert_eq!(
        process.state.health_failures.load(Ordering::SeqCst),
        2,
        "a re-tune never resets the consecutive-failure run"
    );
    assert!(
        !process.is_healthy(),
        "a re-tune never clears the last observation"
    );
    assert!(process.is_alive(), "the child was never restarted");

    assert!(
        tokio::time::timeout(BOUND, process.shutdown(Duration::from_secs(5)))
            .await
            .expect("shutdown must not hang")
            .expect("the fake child acknowledges the shutdown")
    );
}

#[tokio::test]
async fn a_stopped_child_refuses_a_re_tune_instead_of_supervising_a_dead_process() {
    let graceful = process_with(child_config()).await;
    assert!(
        tokio::time::timeout(BOUND, graceful.shutdown(Duration::from_secs(5)))
            .await
            .expect("shutdown must not hang")
            .expect("the fake child acknowledges the shutdown")
    );

    let error = graceful
        .set_health_interval(Duration::from_millis(100))
        .await
        .expect_err("a stopped child has nothing left to supervise");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("has stopped"), "{error}");
    assert!(
        error.to_string().contains("health interval"),
        "the message names what cannot be re-tuned: {error}"
    );
    assert!(
        !health_loop_running(&graceful).await,
        "no loop is started on a dead child"
    );
    assert_eq!(
        graceful
            .set_health_interval(Duration::ZERO)
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::Validation,
        "the refusal is stable, not a one-off"
    );

    // A killed child answers the same.
    let killed = process_with(child_config()).await;
    killed.force_kill().await.expect("the child can be killed");
    let error = killed
        .set_health_interval(Duration::from_millis(100))
        .await
        .expect_err("a killed child cannot be re-tuned either");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("has stopped"), "{error}");
}

#[tokio::test]
async fn the_seam_re_tunes_a_live_child_and_reports_the_stopped_one() {
    let agent: Arc<dyn SpawnedAgent> = tokio::time::timeout(
        BOUND,
        SubprocessSpawner::new(child_config()).spawn(SpawnSpec::class_agent("tests.spawn", "Fake")),
    )
    .await
    .expect("spawn must not hang")
    .expect("the fake child launches");

    tokio::time::timeout(BOUND, agent.set_health_interval(Duration::ZERO))
        .await
        .expect("a re-tune must not hang")
        .expect("the subprocess backend implements re-tuning");
    tokio::time::timeout(BOUND, agent.set_health_interval(Duration::from_millis(150)))
        .await
        .expect("a re-tune must not hang")
        .expect("a live child accepts a re-tune through the seam");
    assert!(agent.is_alive());
    assert!(tokio::time::timeout(BOUND, agent.health())
        .await
        .expect("a probe must not hang")
        .expect("the fake child answers a probe"));
    let output = tokio::time::timeout(BOUND, agent.send(json!({"query": "seam"})))
        .await
        .expect("send must not hang")
        .expect("a re-tuned handle is still in sync");
    assert_eq!(output.output, "pong");

    assert!(
        tokio::time::timeout(BOUND, agent.shutdown(Duration::from_secs(5)))
            .await
            .expect("shutdown must not hang")
            .expect("the fake child acknowledges the shutdown")
    );
    let error = agent
        .set_health_interval(Duration::from_millis(100))
        .await
        .expect_err("a stopped child refuses a re-tune through the seam too");
    assert_eq!(error.kind(), ErrorKind::Validation);
}
