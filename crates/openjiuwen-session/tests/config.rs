//! External-consumer integration tests for [`openjiuwen_session::config::SessionConfig`].
//!
//! Only public paths are used, exactly as downstream workflow/agent crates consume
//! them. Python parity: the defaults mirror `Config._load_builtin_configs_` in
//! `openjiuwen/core/session/config/base.py` (v0.1.20), with the Rust deviations
//! documented on the type itself (millisecond timeouts, case-sensitive booleans).
//!
//! **Environment discipline**: environment variables are process-global, so every
//! test that reads or mutates them holds the file-level [`ENV_LOCK`] through
//! [`EnvGuard`]. The guard clears every managed variable on construction (so
//! assertions never depend on the host environment) and restores the previous
//! values on drop (so a panicking assertion cannot leak a value into another test
//! in the same process). Managed names are the seven canonical keys in
//! [`MANAGED_VARS`].
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::sync::{Mutex, MutexGuard, PoisonError};

use openjiuwen_common::error::ErrorKind;
use openjiuwen_session::config::SessionConfig;
use openjiuwen_session::constants::{
    ENV_COMP_STREAM_CALL_TIMEOUT, ENV_FORCE_DEL_WORKFLOW_STATE, ENV_LOOP_NUMBER_MAX_LIMIT,
    ENV_STREAM_INPUT_GEN_TIMEOUT, ENV_WORKFLOW_EXECUTE_TIMEOUT,
    ENV_WORKFLOW_STREAM_FIRST_FRAME_TIMEOUT, ENV_WORKFLOW_STREAM_FRAME_TIMEOUT,
};
use serde_json::{json, Value};

/// Serializes every test in this binary that touches process environment state.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Every environment variable [`SessionConfig::from_env`] reads.
const MANAGED_VARS: &[&str] = &[
    ENV_WORKFLOW_EXECUTE_TIMEOUT,
    ENV_WORKFLOW_STREAM_FRAME_TIMEOUT,
    ENV_WORKFLOW_STREAM_FIRST_FRAME_TIMEOUT,
    ENV_COMP_STREAM_CALL_TIMEOUT,
    ENV_STREAM_INPUT_GEN_TIMEOUT,
    ENV_LOOP_NUMBER_MAX_LIMIT,
    ENV_FORCE_DEL_WORKFLOW_STATE,
];

/// RAII environment guard: owns [`ENV_LOCK`], clears the managed variables, and
/// restores their previous values on drop.
struct EnvGuard {
    _lock: MutexGuard<'static, ()>,
    saved: Vec<(&'static str, Option<OsString>)>,
}

impl EnvGuard {
    /// Locks the environment and clears every managed variable.
    fn new() -> Self {
        let lock = ENV_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        let saved = MANAGED_VARS
            .iter()
            .map(|name| (*name, std::env::var_os(name)))
            .collect();
        for name in MANAGED_VARS {
            std::env::remove_var(name);
        }
        Self { _lock: lock, saved }
    }

    /// Sets `name` for the lifetime of the guard.
    ///
    /// `&self` ties the mutation to the guard (and therefore to the lock).
    #[allow(clippy::unused_self)]
    fn set(&self, name: &str, value: &str) {
        std::env::set_var(name, value);
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (name, value) in self.saved.drain(..) {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 1. Defaults and serde
// ---------------------------------------------------------------------------

#[test]
fn defaults_match_spec() {
    let config = SessionConfig::default();

    assert!(config.envs.is_empty());
    assert!(config.workflow_configs.is_empty());
    assert_eq!(config.agent_config, None);
    assert_eq!(config.comp_stream_call_timeout_ms, -1);
    assert_eq!(config.stream_input_generator_timeout_ms, -1);
    assert_eq!(config.end_comp_template_branch_render_timeout_ms, 5);
    assert_eq!(config.end_comp_template_render_position_timeout_ms, 5);
    assert_eq!(config.execute_timeout_ms, 60_000);
    assert_eq!(config.stream_frame_timeout_ms, -1);
    assert_eq!(config.stream_first_frame_timeout_ms, -1);
    assert_eq!(config.loop_number_max_limit, 1000);
    assert!(!config.force_del_workflow_state);

    // `new()` reads the environment, so only compare it to `default()` under the
    // lock with every variable cleared; the only difference is the seeded `envs`
    // map, which records the effective defaults.
    let _env = EnvGuard::new();
    let mut from_env = SessionConfig::new();
    assert_eq!(from_env.envs.len(), 7);
    from_env.envs.clear();
    assert_eq!(from_env, SessionConfig::default());
}

#[test]
fn serde_round_trip_preserves_every_field() {
    let mut config = SessionConfig {
        comp_stream_call_timeout_ms: 250,
        stream_input_generator_timeout_ms: 0,
        end_comp_template_branch_render_timeout_ms: 7,
        end_comp_template_render_position_timeout_ms: 8,
        execute_timeout_ms: 90_000,
        stream_frame_timeout_ms: -5,
        stream_first_frame_timeout_ms: 1_500,
        loop_number_max_limit: 42,
        force_del_workflow_state: true,
        ..SessionConfig::default()
    };
    config.set_env("CUSTOM", json!({"nested": [1, 2]}));
    config.set_agent_config(json!({"model": "gpt"}));
    config
        .add_workflow_config("wf", json!({"nodes": []}))
        .expect("valid id");

    let encoded = serde_json::to_value(&config).expect("serializes");
    let decoded: SessionConfig = serde_json::from_value(encoded.clone()).expect("deserializes");
    assert_eq!(decoded, config);

    // Field names are the snake_case JSON contract.
    let object = encoded.as_object().expect("object");
    for key in [
        "envs",
        "workflow_configs",
        "agent_config",
        "comp_stream_call_timeout_ms",
        "stream_input_generator_timeout_ms",
        "end_comp_template_branch_render_timeout_ms",
        "end_comp_template_render_position_timeout_ms",
        "execute_timeout_ms",
        "stream_frame_timeout_ms",
        "stream_first_frame_timeout_ms",
        "loop_number_max_limit",
        "force_del_workflow_state",
    ] {
        assert!(object.contains_key(key), "missing serialized field {key}");
    }
}

#[test]
fn deserialization_defaults_missing_fields_and_ignores_unknown_ones() {
    let empty: SessionConfig = serde_json::from_str("{}").expect("empty object uses defaults");
    assert_eq!(empty, SessionConfig::default());

    let partial: SessionConfig = serde_json::from_str(
        r#"{"execute_timeout_ms": 1234, "force_del_workflow_state": true, "future_field": {"x": 1}}"#,
    )
    .expect("tolerant of unknown fields");
    assert_eq!(partial.execute_timeout_ms, 1234);
    assert!(partial.force_del_workflow_state);
    assert_eq!(
        partial.loop_number_max_limit,
        SessionConfig::default().loop_number_max_limit
    );
}

// ---------------------------------------------------------------------------
// 2. Environment overrides
// ---------------------------------------------------------------------------

#[test]
fn from_env_applies_all_overrides_and_records_raw_values() {
    let env = EnvGuard::new();
    env.set(ENV_WORKFLOW_EXECUTE_TIMEOUT, "45000");
    env.set(ENV_WORKFLOW_STREAM_FRAME_TIMEOUT, "-3");
    env.set(ENV_WORKFLOW_STREAM_FIRST_FRAME_TIMEOUT, "250");
    env.set(ENV_COMP_STREAM_CALL_TIMEOUT, "-7");
    env.set(ENV_STREAM_INPUT_GEN_TIMEOUT, "125");
    env.set(ENV_LOOP_NUMBER_MAX_LIMIT, "17");
    env.set(ENV_FORCE_DEL_WORKFLOW_STATE, "true");

    let config = SessionConfig::from_env();
    assert_eq!(config.execute_timeout_ms, 45_000);
    assert_eq!(config.stream_frame_timeout_ms, -3);
    assert_eq!(config.stream_first_frame_timeout_ms, 250);
    assert_eq!(config.comp_stream_call_timeout_ms, -7);
    assert_eq!(config.stream_input_generator_timeout_ms, 125);
    assert_eq!(config.loop_number_max_limit, 17);
    assert!(config.force_del_workflow_state);

    // Raw environment strings are preserved verbatim in `envs`.
    let envs = config.get_envs();
    assert_eq!(
        envs.get(ENV_WORKFLOW_EXECUTE_TIMEOUT),
        Some(&json!("45000")),
        "raw string, not the parsed integer"
    );
    assert_eq!(
        envs.get(ENV_WORKFLOW_STREAM_FRAME_TIMEOUT),
        Some(&json!("-3"))
    );
    assert_eq!(
        envs.get(ENV_WORKFLOW_STREAM_FIRST_FRAME_TIMEOUT),
        Some(&json!("250"))
    );
    assert_eq!(envs.get(ENV_COMP_STREAM_CALL_TIMEOUT), Some(&json!("-7")));
    assert_eq!(envs.get(ENV_STREAM_INPUT_GEN_TIMEOUT), Some(&json!("125")));
    assert_eq!(envs.get(ENV_LOOP_NUMBER_MAX_LIMIT), Some(&json!("17")));
    assert_eq!(envs.get(ENV_FORCE_DEL_WORKFLOW_STATE), Some(&json!("true")));
    assert_eq!(envs.len(), 7);
}

#[test]
fn from_env_without_variables_records_defaults() {
    let _env = EnvGuard::new();

    let config = SessionConfig::from_env();
    let mut scalar_fields = config.clone();
    scalar_fields.envs.clear();
    assert_eq!(
        scalar_fields,
        SessionConfig::default(),
        "absent variables leave every scalar field at its default"
    );

    let envs = config.get_envs();
    assert_eq!(envs.get(ENV_WORKFLOW_EXECUTE_TIMEOUT), Some(&json!(60_000)));
    assert_eq!(
        envs.get(ENV_WORKFLOW_STREAM_FRAME_TIMEOUT),
        Some(&json!(-1))
    );
    assert_eq!(
        envs.get(ENV_WORKFLOW_STREAM_FIRST_FRAME_TIMEOUT),
        Some(&json!(-1))
    );
    assert_eq!(envs.get(ENV_COMP_STREAM_CALL_TIMEOUT), Some(&json!(-1)));
    assert_eq!(envs.get(ENV_STREAM_INPUT_GEN_TIMEOUT), Some(&json!(-1)));
    assert_eq!(envs.get(ENV_LOOP_NUMBER_MAX_LIMIT), Some(&json!(1000)));
    assert_eq!(envs.get(ENV_FORCE_DEL_WORKFLOW_STATE), Some(&json!(false)));
    assert_eq!(envs.len(), 7);
}

#[test]
fn malformed_values_keep_defaults_but_keep_raw_env_entries() {
    let env = EnvGuard::new();
    env.set(ENV_WORKFLOW_EXECUTE_TIMEOUT, "x");
    env.set(ENV_WORKFLOW_STREAM_FRAME_TIMEOUT, "not-a-number");
    env.set(ENV_WORKFLOW_STREAM_FIRST_FRAME_TIMEOUT, "1.5");
    env.set(ENV_COMP_STREAM_CALL_TIMEOUT, "");
    env.set(ENV_STREAM_INPUT_GEN_TIMEOUT, "  9x ");
    env.set(ENV_LOOP_NUMBER_MAX_LIMIT, "abc");
    env.set(ENV_FORCE_DEL_WORKFLOW_STATE, "yes");

    let config = SessionConfig::from_env();
    let defaults = SessionConfig::default();
    assert_eq!(config.execute_timeout_ms, defaults.execute_timeout_ms);
    assert_eq!(
        config.stream_frame_timeout_ms,
        defaults.stream_frame_timeout_ms
    );
    assert_eq!(
        config.stream_first_frame_timeout_ms,
        defaults.stream_first_frame_timeout_ms
    );
    assert_eq!(
        config.comp_stream_call_timeout_ms,
        defaults.comp_stream_call_timeout_ms
    );
    assert_eq!(
        config.stream_input_generator_timeout_ms,
        defaults.stream_input_generator_timeout_ms
    );
    assert_eq!(config.loop_number_max_limit, defaults.loop_number_max_limit);
    assert_eq!(
        config.force_del_workflow_state,
        defaults.force_del_workflow_state
    );

    let envs = config.get_envs();
    assert_eq!(envs.get(ENV_LOOP_NUMBER_MAX_LIMIT), Some(&json!("abc")));
    assert_eq!(envs.get(ENV_WORKFLOW_EXECUTE_TIMEOUT), Some(&json!("x")));
    assert_eq!(envs.get(ENV_FORCE_DEL_WORKFLOW_STATE), Some(&json!("yes")));
}

#[test]
fn negative_loop_limit_and_timeout_are_rejected() {
    let env = EnvGuard::new();
    env.set(ENV_LOOP_NUMBER_MAX_LIMIT, "-1");
    env.set(ENV_WORKFLOW_EXECUTE_TIMEOUT, "-60000");

    let config = SessionConfig::from_env();
    assert_eq!(config.loop_number_max_limit, 1000);
    assert_eq!(config.execute_timeout_ms, 60_000);
}

#[test]
fn force_del_workflow_state_only_accepts_literal_booleans() {
    let env = EnvGuard::new();

    env.set(ENV_FORCE_DEL_WORKFLOW_STATE, "true");
    assert!(SessionConfig::from_env().force_del_workflow_state);

    env.set(ENV_FORCE_DEL_WORKFLOW_STATE, "false");
    assert!(!SessionConfig::from_env().force_del_workflow_state);

    for spelling in ["TRUE", "False", "yes", "no", "1", "0", ""] {
        env.set(ENV_FORCE_DEL_WORKFLOW_STATE, spelling);
        let config = SessionConfig::from_env();
        assert!(
            !config.force_del_workflow_state,
            "{spelling:?} must keep the default false"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Environment entries, agent config, workflow configs
// ---------------------------------------------------------------------------

#[test]
fn env_entries_are_independent_of_process_environment() {
    let _env = EnvGuard::new();
    let mut config = SessionConfig::default();

    assert_eq!(config.get_env("MISSING"), None);
    config.set_env("A", json!(1));
    config.set_env("B", json!({"k": "v"}));
    assert_eq!(config.get_env("A"), Some(&json!(1)));
    assert_eq!(config.get_env("B"), Some(&json!({"k": "v"})));

    // `set_env` replaces.
    config.set_env("A", json!("replaced"));
    assert_eq!(config.get_env("A"), Some(&json!("replaced")));

    // `get_envs` returns an owned copy: mutating it leaves the config untouched.
    let mut snapshot = config.get_envs();
    assert_eq!(
        snapshot,
        BTreeMap::from([
            ("A".to_string(), json!("replaced")),
            ("B".to_string(), json!({"k": "v"}))
        ])
    );
    snapshot.insert("C".to_string(), json!(3));
    assert_eq!(config.get_env("C"), None);
    assert_eq!(config.get_envs().len(), 2);
}

#[test]
fn agent_config_set_get() {
    let mut config = SessionConfig::default();
    assert_eq!(config.agent_config(), None);

    config.set_agent_config(json!({"model_name": "m"}));
    assert_eq!(config.agent_config(), Some(&json!({"model_name": "m"})));

    config.set_agent_config(json!({"model_name": "m2"}));
    assert_eq!(config.agent_config(), Some(&json!({"model_name": "m2"})));
}

#[test]
fn workflow_configs_insert_replace_and_look_up() {
    let mut config = SessionConfig::default();
    assert_eq!(config.workflow_config("wf"), None);

    config
        .add_workflow_config("wf", json!({"version": 1}))
        .expect("valid id");
    assert_eq!(config.workflow_config("wf"), Some(&json!({"version": 1})));

    // Insert/replace semantics.
    config
        .add_workflow_config(String::from("wf"), json!({"version": 2}))
        .expect("valid id");
    assert_eq!(config.workflow_config("wf"), Some(&json!({"version": 2})));
    assert_eq!(config.workflow_configs.len(), 1);

    config
        .add_workflow_config("other", json!({}))
        .expect("valid id");
    assert_eq!(config.workflow_configs.len(), 2);
    assert_eq!(config.workflow_config("missing"), None);

    // Empty and whitespace-only ids are validation errors (Python raises
    // `ValueError` for a None id).
    for bad in ["", " ", "\t\n"] {
        let error = config
            .add_workflow_config(bad, json!({}))
            .expect_err("blank id rejected");
        assert_eq!(error.kind(), ErrorKind::Validation);
    }
    assert_eq!(
        config.workflow_configs.len(),
        2,
        "rejected ids are not inserted"
    );
}

#[test]
fn config_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SessionConfig>();
    let value: Value = json!({});
    assert!(value.is_object());
}
