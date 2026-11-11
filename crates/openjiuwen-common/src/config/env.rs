//! Environment variable resolvers.
//!
//! Python reads these knobs inline with `os.getenv(...)`; Rust funnels them
//! through this module so the variable names, the trimming rules, and the
//! defaults are defined exactly once.
//!
//! All resolvers are pure readers: they never write to the process environment
//! and never panic. [`load_dotenv`](super::load_dotenv) is the only function in
//! the crate that mutates it, and it belongs to binaries and examples.

use std::path::PathBuf;

use crate::error::{Error, Result};
use crate::security::SecretString;

/// Root directory for runtime state (logs, sessions): `OPENJIUWEN_HOME`.
pub const OPENJIUWEN_HOME: &str = "OPENJIUWEN_HOME";

/// Primary `openJiuwen` API key variable: `OPENJIUWEN_API_KEY`.
pub const OPENJIUWEN_API_KEY: &str = "OPENJIUWEN_API_KEY";

/// `openJiuwen` server base URL: `OPENJIUWEN_SERVER_URL`.
pub const OPENJIUWEN_SERVER_URL: &str = "OPENJIUWEN_SERVER_URL";

/// Shell strictness switch: `OPENJIUWEN_BASH_STRICT`.
pub const OPENJIUWEN_BASH_STRICT: &str = "OPENJIUWEN_BASH_STRICT";

/// MCP reconnect attempt count: `OPENJIUWEN_MCP_RECONNECT_ATTEMPTS`.
pub const OPENJIUWEN_MCP_RECONNECT_ATTEMPTS: &str = "OPENJIUWEN_MCP_RECONNECT_ATTEMPTS";

/// Global OpenTelemetry instrumentation switch:
/// `OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE`.
pub const OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE: &str =
    "OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE";

/// Log level override: `OPENJIUWEN_LOG_LEVEL`.
pub const OPENJIUWEN_LOG_LEVEL: &str = "OPENJIUWEN_LOG_LEVEL";

/// Log format override (`text`/`json`): `OPENJIUWEN_LOG_FORMAT`.
pub const OPENJIUWEN_LOG_FORMAT: &str = "OPENJIUWEN_LOG_FORMAT";

/// Fallback API key variable shared with other SDKs: `API_KEY`.
pub const API_KEY: &str = "API_KEY";

/// Fallback API base URL shared with other SDKs: `API_BASE`.
pub const API_BASE: &str = "API_BASE";

/// Model provider name: `MODEL_PROVIDER`.
pub const MODEL_PROVIDER: &str = "MODEL_PROVIDER";

/// Model name: `MODEL_NAME`.
pub const MODEL_NAME: &str = "MODEL_NAME";

/// OpenAI-compatible API key variable: `OPENAI_API_KEY`.
pub const OPENAI_API_KEY: &str = "OPENAI_API_KEY";

/// OpenAI-compatible base URL variable: `OPENAI_BASE_URL`.
pub const OPENAI_BASE_URL: &str = "OPENAI_BASE_URL";

/// SSRF protection switch: `SSRF_PROTECT_ENABLED`.
pub const SSRF_PROTECT_ENABLED: &str = "SSRF_PROTECT_ENABLED";

/// Default model provider when `MODEL_PROVIDER` is unset: `"openai"`.
pub const DEFAULT_MODEL_PROVIDER: &str = "openai";

/// Default MCP reconnect attempts when the variable is unset or unparsable: `3`.
pub const DEFAULT_MCP_RECONNECT_ATTEMPTS: u32 = 3;

/// Reads and normalizes an environment variable.
///
/// The value is trimmed of surrounding whitespace. An unset, empty,
/// whitespace-only, or non-UTF-8 value yields `None`, so callers never have to
/// guard against blank strings (Python parity: `os.getenv(name) or None`).
#[must_use]
pub fn var(name: &str) -> Option<String> {
    let raw = std::env::var(name).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

/// Interprets a string as a boolean flag, case-insensitively.
///
/// `"1"`, `"true"`, `"yes"`, and `"on"` (in any ASCII case, surrounding
/// whitespace ignored) are `true`; every other value — including `"0"`,
/// `"false"`, `"no"`, `"off"`, and `""` — is `false` (Python parity).
#[must_use]
pub fn truthy(value: &str) -> bool {
    let value = value.trim();
    ["1", "true", "yes", "on"]
        .iter()
        .any(|candidate| value.eq_ignore_ascii_case(candidate))
}

/// Reads a boolean flag: [`var`] followed by [`truthy`].
///
/// Returns `None` when the variable is absent or blank, so callers can apply
/// their own default instead of guessing.
#[must_use]
pub fn flag(name: &str) -> Option<bool> {
    var(name).map(|value| truthy(&value))
}

/// Reads and parses an environment variable.
///
/// Returns `Ok(None)` when the variable is absent or blank. The value is trimmed
/// before parsing, so ` 5 ` parses as `5`.
///
/// # Errors
///
/// Returns [`Error::Config`] naming the variable when the value does not parse
/// as `T`.
pub fn parse<T>(name: &str) -> Result<Option<T>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    var(name)
        .map(|raw| {
            raw.parse::<T>()
                .map_err(|err| Error::config(format!("invalid value for {name}: {err}")))
        })
        .transpose()
}

/// Returns the first non-empty value among `names`.
fn first_of(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| var(name))
}

/// Resolves the LLM API key.
///
/// Precedence: `OPENJIUWEN_API_KEY` → `API_KEY` → `OPENAI_API_KEY`. The value is
/// wrapped in a [`SecretString`], so it is redacted in `Debug` output and logs.
#[must_use]
pub fn api_key() -> Option<SecretString> {
    first_of(&[OPENJIUWEN_API_KEY, API_KEY, OPENAI_API_KEY]).map(SecretString::new)
}

/// Resolves the LLM API base URL.
///
/// Precedence: `API_BASE` → `OPENAI_BASE_URL`. No default is applied here; the
/// provider client owns its own default endpoint.
#[must_use]
pub fn api_base() -> Option<String> {
    first_of(&[API_BASE, OPENAI_BASE_URL])
}

/// Resolves the model provider name.
///
/// Reads `MODEL_PROVIDER`, defaulting to [`DEFAULT_MODEL_PROVIDER`] (`"openai"`).
#[must_use]
pub fn model_provider() -> String {
    var(MODEL_PROVIDER).unwrap_or_else(|| DEFAULT_MODEL_PROVIDER.to_owned())
}

/// Resolves the model name from `MODEL_NAME`.
#[must_use]
pub fn model_name() -> Option<String> {
    var(MODEL_NAME)
}

/// Resolves the `openJiuwen` home directory from `OPENJIUWEN_HOME`.
///
/// Use [`OpenJiuwenHome::resolve`](crate::config::OpenJiuwenHome::resolve) when
/// the `$HOME/.openjiuwen` and relative fallbacks are also wanted.
#[must_use]
pub fn openjiuwen_home() -> Option<PathBuf> {
    var(OPENJIUWEN_HOME).map(PathBuf::from)
}

/// Resolves the `openJiuwen` server URL from `OPENJIUWEN_SERVER_URL`.
#[must_use]
pub fn server_url() -> Option<String> {
    var(OPENJIUWEN_SERVER_URL)
}

/// Resolves whether shell commands run in strict mode.
///
/// Reads `OPENJIUWEN_BASH_STRICT` through [`flag`]; an absent or unrecognized
/// value means `false` (Python default).
#[must_use]
pub fn bash_strict() -> bool {
    flag(OPENJIUWEN_BASH_STRICT).unwrap_or(false)
}

/// Resolves the MCP reconnect attempt count.
///
/// Reads `OPENJIUWEN_MCP_RECONNECT_ATTEMPTS`, defaulting to
/// [`DEFAULT_MCP_RECONNECT_ATTEMPTS`] (`3`) when the variable is absent, blank,
/// or not a valid `u32`. A bad value is tolerated here rather than rejected so a
/// typo cannot prevent an agent from starting.
#[must_use]
pub fn mcp_reconnect_attempts() -> u32 {
    parse::<u32>(OPENJIUWEN_MCP_RECONNECT_ATTEMPTS)
        .ok()
        .flatten()
        .unwrap_or(DEFAULT_MCP_RECONNECT_ATTEMPTS)
}

/// Resolves whether global OpenTelemetry instrumentation is enabled.
///
/// Reads `OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE` through [`flag`]; default
/// `false`.
#[must_use]
pub fn otel_global_instrument_enabled() -> bool {
    flag(OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE).unwrap_or(false)
}

/// Resolves whether outbound URL checks block SSRF targets.
///
/// Reads `SSRF_PROTECT_ENABLED` through [`flag`]; default `true`, so protection
/// is on unless explicitly disabled (Python default).
#[must_use]
pub fn ssrf_protect_enabled() -> bool {
    flag(SSRF_PROTECT_ENABLED).unwrap_or(true)
}

/// Resolves the log level override from `OPENJIUWEN_LOG_LEVEL`.
///
/// Returns the raw value; normalization to a `tracing` level is the logging
/// subsystem's job.
#[must_use]
pub fn log_level() -> Option<String> {
    var(OPENJIUWEN_LOG_LEVEL)
}

/// Resolves the log format override from `OPENJIUWEN_LOG_FORMAT`.
///
/// Returns the raw value (`text`/`json`); the logging subsystem validates it.
#[must_use]
pub fn log_format() -> Option<String> {
    var(OPENJIUWEN_LOG_FORMAT)
}

/// Resolves the outbound HTTP proxy.
///
/// Precedence: `http_proxy` → `https_proxy` → `HTTP_PROXY` → `HTTPS_PROXY`
/// (lowercase first, as the conventional curl-style lookup does).
#[must_use]
pub fn http_proxy() -> Option<String> {
    first_of(&["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"])
}

/// Resolves the proxy bypass list.
///
/// Precedence: `NO_PROXY` → `no_proxy` (uppercase first, matching the
/// `reqwest`/`curl` convention).
#[must_use]
pub fn no_proxy() -> Option<String> {
    first_of(&["NO_PROXY", "no_proxy"])
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::config::test_support::EnvGuard;
    use crate::error::ErrorKind;

    // Python parity: the Python SDK reads these names with inline
    // `os.getenv(...)` calls; the assertions below pin the same lookup rules.

    #[test]
    fn var_trims_and_treats_blank_as_absent() {
        let env = EnvGuard::new(&[OPENJIUWEN_API_KEY]);

        env.set(OPENJIUWEN_API_KEY, "  sk-abc  ");
        assert_eq!(var(OPENJIUWEN_API_KEY).as_deref(), Some("sk-abc"));

        env.set(OPENJIUWEN_API_KEY, "   ");
        assert_eq!(var(OPENJIUWEN_API_KEY), None);

        env.set(OPENJIUWEN_API_KEY, "");
        assert_eq!(var(OPENJIUWEN_API_KEY), None);

        env.remove(OPENJIUWEN_API_KEY);
        assert_eq!(var(OPENJIUWEN_API_KEY), None);
    }

    #[cfg(unix)]
    #[test]
    fn var_returns_none_for_invalid_utf8() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let _env = EnvGuard::new(&[OPENJIUWEN_API_KEY]);
        std::env::set_var(
            OPENJIUWEN_API_KEY,
            OsString::from_vec(b"sk-\xff-bad".to_vec()),
        );
        assert_eq!(var(OPENJIUWEN_API_KEY), None);
    }

    #[test]
    fn truthy_accepts_python_variants() {
        for value in [
            "1", "true", "TRUE", "True", "yes", "YES", "on", "On", " on ",
        ] {
            assert!(truthy(value), "{value:?} should be truthy");
        }
        for value in ["", " ", "0", "false", "no", "off", "2", "enabled", "y"] {
            assert!(!truthy(value), "{value:?} should be falsy");
        }
    }

    #[test]
    fn flag_maps_absent_and_present_values() {
        let env = EnvGuard::new(&[OPENJIUWEN_BASH_STRICT]);
        assert_eq!(flag(OPENJIUWEN_BASH_STRICT), None);
        env.set(OPENJIUWEN_BASH_STRICT, "yes");
        assert_eq!(flag(OPENJIUWEN_BASH_STRICT), Some(true));
        env.set(OPENJIUWEN_BASH_STRICT, "0");
        assert_eq!(flag(OPENJIUWEN_BASH_STRICT), Some(false));
    }

    #[test]
    fn api_key_precedence_drops_one_variable_at_a_time() {
        let env = EnvGuard::new(&[OPENJIUWEN_API_KEY, API_KEY, OPENAI_API_KEY]);
        assert_eq!(api_key(), None);

        env.set(OPENJIUWEN_API_KEY, "jiuwen-key");
        env.set(API_KEY, "api-key");
        env.set(OPENAI_API_KEY, "openai-key");
        assert_eq!(api_key(), Some(SecretString::new("jiuwen-key")));

        env.remove(OPENJIUWEN_API_KEY);
        assert_eq!(api_key(), Some(SecretString::new("api-key")));

        env.remove(API_KEY);
        assert_eq!(api_key(), Some(SecretString::new("openai-key")));

        env.remove(OPENAI_API_KEY);
        assert_eq!(api_key(), None);
    }

    #[test]
    fn api_key_skips_blank_values() {
        let env = EnvGuard::new(&[OPENJIUWEN_API_KEY, API_KEY, OPENAI_API_KEY]);
        env.set(OPENJIUWEN_API_KEY, "  ");
        env.set(OPENAI_API_KEY, "openai-key");
        assert_eq!(api_key(), Some(SecretString::new("openai-key")));
    }

    #[test]
    fn api_base_precedence() {
        let env = EnvGuard::new(&[API_BASE, OPENAI_BASE_URL]);
        assert_eq!(api_base(), None);

        env.set(API_BASE, "https://api.jiuwen.example/v1");
        env.set(OPENAI_BASE_URL, "https://api.openai.example/v1");
        assert_eq!(api_base().as_deref(), Some("https://api.jiuwen.example/v1"));

        env.remove(API_BASE);
        assert_eq!(api_base().as_deref(), Some("https://api.openai.example/v1"));

        env.remove(OPENAI_BASE_URL);
        assert_eq!(api_base(), None);
    }

    #[test]
    fn defaults_match_python_values() {
        let env = EnvGuard::new(&[
            MODEL_PROVIDER,
            OPENJIUWEN_BASH_STRICT,
            OPENJIUWEN_MCP_RECONNECT_ATTEMPTS,
            OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE,
            SSRF_PROTECT_ENABLED,
        ]);

        assert_eq!(model_provider(), "openai");
        assert!(!bash_strict());
        assert_eq!(mcp_reconnect_attempts(), 3);
        assert!(!otel_global_instrument_enabled());
        assert!(ssrf_protect_enabled());

        env.set(MODEL_PROVIDER, "anthropic");
        env.set(OPENJIUWEN_BASH_STRICT, "TRUE");
        env.set(OPENJIUWEN_MCP_RECONNECT_ATTEMPTS, " 7 ");
        env.set(OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE, "yes");
        env.set(SSRF_PROTECT_ENABLED, "off");

        assert_eq!(model_provider(), "anthropic");
        assert!(bash_strict());
        assert_eq!(mcp_reconnect_attempts(), 7);
        assert!(otel_global_instrument_enabled());
        assert!(!ssrf_protect_enabled());
    }

    #[test]
    fn parse_reports_invalid_values_and_trims_valid_ones() {
        let env = EnvGuard::new(&[OPENJIUWEN_MCP_RECONNECT_ATTEMPTS]);

        assert_eq!(
            parse::<u32>(OPENJIUWEN_MCP_RECONNECT_ATTEMPTS).unwrap(),
            None
        );

        env.set(OPENJIUWEN_MCP_RECONNECT_ATTEMPTS, " 5 ");
        assert_eq!(
            parse::<u32>(OPENJIUWEN_MCP_RECONNECT_ATTEMPTS).unwrap(),
            Some(5)
        );

        env.set(OPENJIUWEN_MCP_RECONNECT_ATTEMPTS, "5x");
        let err = parse::<u32>(OPENJIUWEN_MCP_RECONNECT_ATTEMPTS).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Config);
        let message = err.to_string();
        assert!(message.contains(OPENJIUWEN_MCP_RECONNECT_ATTEMPTS));
        assert!(message.contains("invalid value for"));
    }

    #[test]
    fn invalid_reconnect_attempts_falls_back_to_default() {
        let env = EnvGuard::new(&[OPENJIUWEN_MCP_RECONNECT_ATTEMPTS]);
        env.set(OPENJIUWEN_MCP_RECONNECT_ATTEMPTS, "many");
        assert_eq!(mcp_reconnect_attempts(), DEFAULT_MCP_RECONNECT_ATTEMPTS);
        env.set(OPENJIUWEN_MCP_RECONNECT_ATTEMPTS, "-1");
        assert_eq!(mcp_reconnect_attempts(), DEFAULT_MCP_RECONNECT_ATTEMPTS);
    }

    #[test]
    fn proxy_precedence() {
        let env = EnvGuard::new(&[
            "http_proxy",
            "https_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "no_proxy",
        ]);

        assert_eq!(http_proxy(), None);
        assert_eq!(no_proxy(), None);

        env.set("HTTP_PROXY", "http://upper.example:8080");
        env.set("HTTPS_PROXY", "http://upper-tls.example:8080");
        env.set("https_proxy", "http://lower-tls.example:8080");
        assert_eq!(
            http_proxy().as_deref(),
            Some("http://lower-tls.example:8080")
        );

        env.set("http_proxy", "http://lower.example:8080");
        assert_eq!(http_proxy().as_deref(), Some("http://lower.example:8080"));

        env.remove("http_proxy");
        env.remove("https_proxy");
        assert_eq!(http_proxy().as_deref(), Some("http://upper.example:8080"));

        env.set("NO_PROXY", "upper.internal");
        env.set("no_proxy", "lower.internal");
        assert_eq!(no_proxy().as_deref(), Some("upper.internal"));

        env.remove("NO_PROXY");
        assert_eq!(no_proxy().as_deref(), Some("lower.internal"));
    }

    #[test]
    fn remaining_resolvers_read_their_variables() {
        let env = EnvGuard::new(&[
            MODEL_NAME,
            OPENJIUWEN_SERVER_URL,
            OPENJIUWEN_LOG_LEVEL,
            OPENJIUWEN_LOG_FORMAT,
            OPENJIUWEN_HOME,
        ]);

        assert_eq!(model_name(), None);
        assert_eq!(server_url(), None);
        assert_eq!(log_level(), None);
        assert_eq!(log_format(), None);
        assert_eq!(openjiuwen_home(), None);

        env.set(MODEL_NAME, "gpt-4o");
        env.set(OPENJIUWEN_SERVER_URL, "http://127.0.0.1:8080");
        env.set(OPENJIUWEN_LOG_LEVEL, "debug");
        env.set(OPENJIUWEN_LOG_FORMAT, "json");
        env.set(OPENJIUWEN_HOME, "env-home");

        assert_eq!(model_name().as_deref(), Some("gpt-4o"));
        assert_eq!(server_url().as_deref(), Some("http://127.0.0.1:8080"));
        assert_eq!(log_level().as_deref(), Some("debug"));
        assert_eq!(log_format().as_deref(), Some("json"));
        assert_eq!(openjiuwen_home(), Some(PathBuf::from("env-home")));
    }
}
