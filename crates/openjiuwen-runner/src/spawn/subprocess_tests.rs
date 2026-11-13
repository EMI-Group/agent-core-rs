#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
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
