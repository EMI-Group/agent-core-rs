//! External-consumer integration tests for configuration and home resolution.
//!
//! Only public paths are used (`openjiuwen_common::config::{env, OpenJiuwenHome}`
//! and `openjiuwen_common::security::SecretString`), exactly as downstream
//! crates and the CLI consume them.
//!
//! **Environment discipline**: environment variables are process-global, so
//! every test that reads or mutates them takes the file-level [`ENV_LOCK`] via
//! [`EnvGuard`]. The guard clears every managed variable on construction (so
//! assertions never depend on the host environment) and restores the previous
//! values in `Drop` (so a panicking assertion cannot leak a value into another
//! test in the same process). Managed names are listed in [`MANAGED_VARS`].
//!
//! Python parity: the Python SDK reads these knobs inline with `os.getenv(...)`
//! (`openjiuwen/core/common/`); the assertions pin the same lookup order,
//! trimming rules, and defaults documented in `crates/openjiuwen-common/CONTEXT.md`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use openjiuwen_common::config::{env, OpenJiuwenHome};
use openjiuwen_common::error::ErrorKind;
use openjiuwen_common::security::SecretString;

/// Serializes every test in this binary that touches process environment state.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Every environment variable these tests read or write, so a test can start
/// from a fully deterministic process environment.
const MANAGED_VARS: &[&str] = &[
    "OPENJIUWEN_API_KEY",
    "API_KEY",
    "OPENAI_API_KEY",
    "API_BASE",
    "OPENAI_BASE_URL",
    "MODEL_PROVIDER",
    "MODEL_NAME",
    "OPENJIUWEN_BASH_STRICT",
    "OPENJIUWEN_MCP_RECONNECT_ATTEMPTS",
    "SSRF_PROTECT_ENABLED",
    "OPENJIUWEN_HOME",
    "OPENJIUWEN_SERVER_URL",
    "OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE",
    "OPENJIUWEN_LOG_LEVEL",
    "OPENJIUWEN_LOG_FORMAT",
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

    /// Removes `name` for the lifetime of the guard.
    #[allow(clippy::unused_self)]
    fn remove(&self, name: &str) {
        std::env::remove_var(name);
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
// 1. API key precedence and `SecretString` behavior
// ---------------------------------------------------------------------------

#[test]
fn api_key_precedence_falls_back_one_variable_at_a_time() {
    let env = EnvGuard::new();

    assert_eq!(env::api_key(), None, "all variables cleared");

    env.set(env::OPENJIUWEN_API_KEY, "sk-jiuwen");
    env.set(env::API_KEY, "sk-api");
    env.set(env::OPENAI_API_KEY, "sk-openai");
    let key = env::api_key().unwrap();
    assert_eq!(key.expose_secret(), "sk-jiuwen");

    env.remove(env::OPENJIUWEN_API_KEY);
    assert_eq!(env::api_key().unwrap().expose_secret(), "sk-api");

    env.remove(env::API_KEY);
    assert_eq!(env::api_key().unwrap().expose_secret(), "sk-openai");

    env.remove(env::OPENAI_API_KEY);
    assert_eq!(env::api_key(), None);
}

#[test]
fn api_key_is_a_redacted_secret_that_still_serializes() {
    let env = EnvGuard::new();
    env.set(env::OPENJIUWEN_API_KEY, "sk-test");

    let key = env::api_key().unwrap();
    assert_eq!(key.expose_secret(), "sk-test");
    assert_eq!(key.len(), "sk-test".len());
    assert!(!key.is_empty());

    let debug = format!("{key:?}");
    assert!(debug.contains("REDACTED"), "debug output: {debug}");
    assert!(!debug.contains("sk-test"), "debug output leaked: {debug}");

    assert_eq!(key.to_string(), "<REDACTED>");
    assert_eq!(format!("{key}"), "<REDACTED>");

    // Config round-trips keep the plaintext (Python parity: secrets are part of
    // the serialized configuration).
    assert_eq!(serde_json::to_string(&key).unwrap(), "\"sk-test\"");

    // An empty secret reports itself as empty rather than as a usable key.
    let empty = SecretString::new("");
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
}

#[test]
fn blank_api_key_values_are_treated_as_absent() {
    let env = EnvGuard::new();

    env.set(env::OPENJIUWEN_API_KEY, "   ");
    env.set(env::API_KEY, "sk-api");
    assert_eq!(env::api_key().unwrap().expose_secret(), "sk-api");

    env.set(env::OPENJIUWEN_API_KEY, "");
    assert_eq!(env::api_key().unwrap().expose_secret(), "sk-api");

    env.set(env::API_KEY, " \t ");
    env.remove(env::OPENAI_API_KEY);
    assert_eq!(env::api_key(), None, "all blank variables are absent");

    // A present value is trimmed before it becomes a secret.
    env.set(env::OPENJIUWEN_API_KEY, "  sk-padded  ");
    assert_eq!(env::api_key().unwrap().expose_secret(), "sk-padded");
}

// ---------------------------------------------------------------------------
// 2. API base precedence
// ---------------------------------------------------------------------------

#[test]
fn api_base_prefers_api_base_over_openai_base_url() {
    let env = EnvGuard::new();

    assert_eq!(env::api_base(), None);

    env.set(env::OPENAI_BASE_URL, "https://openai.example/v1");
    assert_eq!(
        env::api_base().as_deref(),
        Some("https://openai.example/v1")
    );

    env.set(env::API_BASE, "https://api-base.example/v1");
    assert_eq!(
        env::api_base().as_deref(),
        Some("https://api-base.example/v1")
    );

    env.remove(env::API_BASE);
    assert_eq!(
        env::api_base().as_deref(),
        Some("https://openai.example/v1")
    );

    env.remove(env::OPENAI_BASE_URL);
    assert_eq!(env::api_base(), None);
}

// ---------------------------------------------------------------------------
// 3. Defaults and overrides
// ---------------------------------------------------------------------------

#[test]
fn model_provider_defaults_to_openai_and_honors_override() {
    let env = EnvGuard::new();

    assert_eq!(env::model_provider(), env::DEFAULT_MODEL_PROVIDER);
    assert_eq!(env::model_provider(), "openai");

    env.set(env::MODEL_PROVIDER, "anthropic");
    assert_eq!(env::model_provider(), "anthropic");

    // A blank override falls back to the default rather than yielding "".
    env.set(env::MODEL_PROVIDER, "  ");
    assert_eq!(env::model_provider(), "openai");
}

#[test]
fn bash_strict_defaults_to_false_and_accepts_truthy_values() {
    let env = EnvGuard::new();

    assert!(!env::bash_strict());

    for value in ["1", "true", "yes", "ON", "True", "YeS", " on "] {
        env.set(env::OPENJIUWEN_BASH_STRICT, value);
        assert!(env::bash_strict(), "{value:?} should enable strict mode");
    }

    for value in ["0", "false", "no", "off", "OFF", "", "maybe"] {
        env.set(env::OPENJIUWEN_BASH_STRICT, value);
        assert!(
            !env::bash_strict(),
            "{value:?} should not enable strict mode"
        );
    }
}

#[test]
fn mcp_reconnect_attempts_defaults_overrides_and_tolerates_garbage() {
    let env = EnvGuard::new();

    assert_eq!(env::mcp_reconnect_attempts(), 3);
    assert_eq!(
        env::mcp_reconnect_attempts(),
        env::DEFAULT_MCP_RECONNECT_ATTEMPTS
    );

    env.set(env::OPENJIUWEN_MCP_RECONNECT_ATTEMPTS, "7");
    assert_eq!(env::mcp_reconnect_attempts(), 7);

    // Invalid values must not panic or fail: they fall back to the default so a
    // typo cannot prevent an agent from starting.
    for value in ["not-a-number", "-1", "3.5", ""] {
        env.set(env::OPENJIUWEN_MCP_RECONNECT_ATTEMPTS, value);
        assert_eq!(
            env::mcp_reconnect_attempts(),
            3,
            "{value:?} should fall back to the default"
        );
    }
}

#[test]
fn ssrf_protection_defaults_to_enabled_and_can_be_disabled() {
    let env = EnvGuard::new();

    assert!(env::ssrf_protect_enabled(), "protection is on by default");

    for value in ["0", "false", "no", "off", "FALSE"] {
        env.set(env::SSRF_PROTECT_ENABLED, value);
        assert!(
            !env::ssrf_protect_enabled(),
            "{value:?} should disable protection"
        );
    }

    for value in ["1", "true", "yes", "on"] {
        env.set(env::SSRF_PROTECT_ENABLED, value);
        assert!(
            env::ssrf_protect_enabled(),
            "{value:?} should keep protection enabled"
        );
    }
}

#[test]
fn remaining_resolvers_read_their_variables() {
    let env = EnvGuard::new();

    assert_eq!(env::model_name(), None);
    assert_eq!(env::server_url(), None);
    assert_eq!(env::log_level(), None);
    assert_eq!(env::log_format(), None);
    assert_eq!(env::openjiuwen_home(), None);
    assert!(!env::otel_global_instrument_enabled());

    env.set(env::MODEL_NAME, "gpt-4o");
    env.set(env::OPENJIUWEN_SERVER_URL, "http://127.0.0.1:8080");
    env.set(env::OPENJIUWEN_LOG_LEVEL, "debug");
    env.set(env::OPENJIUWEN_LOG_FORMAT, "json");
    env.set(env::OPENJIUWEN_HOME, "env-home");
    env.set(env::OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE, "true");

    assert_eq!(env::model_name().as_deref(), Some("gpt-4o"));
    assert_eq!(env::server_url().as_deref(), Some("http://127.0.0.1:8080"));
    assert_eq!(env::log_level().as_deref(), Some("debug"));
    assert_eq!(env::log_format().as_deref(), Some("json"));
    assert_eq!(env::openjiuwen_home(), Some(PathBuf::from("env-home")));
    assert!(env::otel_global_instrument_enabled());
}

// ---------------------------------------------------------------------------
// 4. Primitive readers: `var`, `flag`, `truthy`, `parse`
// ---------------------------------------------------------------------------

#[test]
fn var_trims_surrounding_whitespace_and_treats_blank_as_absent() {
    let env = EnvGuard::new();

    env.set(env::OPENJIUWEN_API_KEY, "  value  ");
    assert_eq!(env::var(env::OPENJIUWEN_API_KEY).as_deref(), Some("value"));

    env.set(env::OPENJIUWEN_API_KEY, "   ");
    assert_eq!(env::var(env::OPENJIUWEN_API_KEY), None);

    env.set(env::OPENJIUWEN_API_KEY, "");
    assert_eq!(env::var(env::OPENJIUWEN_API_KEY), None);

    env.remove(env::OPENJIUWEN_API_KEY);
    assert_eq!(env::var(env::OPENJIUWEN_API_KEY), None);
    assert_eq!(env::var("OPENJIUWEN_DEFINITELY_UNSET_VARIABLE"), None);
}

#[test]
fn flag_maps_absent_and_present_values() {
    let env = EnvGuard::new();

    assert_eq!(env::flag(env::OPENJIUWEN_BASH_STRICT), None);
    env.set(env::OPENJIUWEN_BASH_STRICT, "0");
    assert_eq!(env::flag(env::OPENJIUWEN_BASH_STRICT), Some(false));
    env.set(env::OPENJIUWEN_BASH_STRICT, "yes");
    assert_eq!(env::flag(env::OPENJIUWEN_BASH_STRICT), Some(true));
    env.set(env::OPENJIUWEN_BASH_STRICT, "  ");
    assert_eq!(env::flag(env::OPENJIUWEN_BASH_STRICT), None);
}

#[test]
fn truthy_accepts_python_variants_case_insensitively() {
    for value in [
        "1", "true", "TRUE", "True", "yes", "YES", "Yes", "on", "On", "ON",
    ] {
        assert!(env::truthy(value), "{value:?} should be truthy");
    }
    for value in ["0", "false", "NO", "off", "", " ", "maybe", "2"] {
        assert!(!env::truthy(value), "{value:?} should be falsy");
    }
}

#[test]
fn parse_returns_none_some_and_a_config_error() {
    let env = EnvGuard::new();

    assert_eq!(
        env::parse::<u32>(env::OPENJIUWEN_MCP_RECONNECT_ATTEMPTS).unwrap(),
        None
    );

    env.set(env::OPENJIUWEN_MCP_RECONNECT_ATTEMPTS, " 5 ");
    assert_eq!(
        env::parse::<u32>(env::OPENJIUWEN_MCP_RECONNECT_ATTEMPTS).unwrap(),
        Some(5)
    );

    env.set(env::OPENJIUWEN_MCP_RECONNECT_ATTEMPTS, "garbage");
    let err = env::parse::<u32>(env::OPENJIUWEN_MCP_RECONNECT_ATTEMPTS).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Config);
    assert!(err
        .to_string()
        .contains(env::OPENJIUWEN_MCP_RECONNECT_ATTEMPTS));
}

// ---------------------------------------------------------------------------
// 5. `OpenJiuwenHome` resolution and creation
// ---------------------------------------------------------------------------

#[test]
fn resolve_honors_openjiuwen_home_environment_variable() {
    let env = EnvGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("env-home");
    let root_str = root.to_str().unwrap().to_owned();

    env.set(env::OPENJIUWEN_HOME, &root_str);

    assert_eq!(env::openjiuwen_home(), Some(PathBuf::from(&root_str)));
    assert_eq!(OpenJiuwenHome::resolve().root(), root.as_path());
    // `Default` delegates to `resolve`, so it follows the same variable.
    assert_eq!(OpenJiuwenHome::default().root(), root.as_path());
}

#[test]
fn from_root_derives_join_logs_and_sessions_paths() {
    let home = OpenJiuwenHome::from_root("/var/lib/openjiuwen");

    assert_eq!(home.root(), Path::new("/var/lib/openjiuwen"));
    assert_eq!(
        home.join("x"),
        PathBuf::from("/var/lib/openjiuwen/x"),
        "join appends a relative path"
    );
    assert_eq!(home.logs_dir(), PathBuf::from("/var/lib/openjiuwen/logs"));
    assert_eq!(
        home.sessions_dir(),
        PathBuf::from("/var/lib/openjiuwen/sessions")
    );
}

#[test]
fn ensure_creates_only_the_root_and_is_idempotent() {
    let _env = EnvGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let home = OpenJiuwenHome::from_root(dir.path().join("home"));

    assert!(!home.root().exists(), "constructing a home does no I/O");

    assert!(home.ensure().is_ok());
    assert!(home.root().exists());
    assert!(home.root().is_dir());

    // Idempotent: creating an existing root is success.
    assert!(home.ensure().is_ok());
    assert!(home.root().is_dir());

    // Documented behavior: `ensure()` creates the root only, so `logs/` is a
    // derived path that callers create when they need it.
    assert_eq!(home.logs_dir(), home.root().join("logs"));
    assert!(!home.logs_dir().exists(), "ensure() does not create logs/");
    std::fs::create_dir_all(home.logs_dir()).unwrap();
    assert!(home.logs_dir().is_dir());

    // The sessions directory is likewise derived, not created by `ensure()`.
    std::fs::create_dir_all(home.sessions_dir()).unwrap();
    assert!(home.sessions_dir().is_dir());
}

#[test]
fn ensure_reports_config_errors_instead_of_panicking() {
    let _env = EnvGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let blocking = dir.path().join("blocking-file");
    std::fs::write(&blocking, b"not a directory").unwrap();

    let home = OpenJiuwenHome::from_root(blocking.join("nested"));
    let err = home.ensure().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Config);
}
