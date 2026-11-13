//! Process invocation of the DeepSeek Harness transport: `DSH_HOME`, argv and environment.
//!
//! The transport contract these helpers implement is documented on the
//! [`harness`](super::harness) module; nothing here spawns a process or talks to the runtime.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_harness_protocol::{HarnessContext, HarnessError, HarnessResult};

use super::DshHarnessConfig;

/// Environment variable locating the DSH home directory.
pub(super) const DSH_HOME_ENV: &str = "DSH_HOME";

/// Environment variable injecting an external endpoint base URL.
const DEEPSEEK_BASE_URL_ENV: &str = "DEEPSEEK_BASE_URL";

/// Environment variable injecting an external endpoint API key.
const DEEPSEEK_API_KEY_ENV: &str = "DEEPSEEK_API_KEY";

/// Message reported when neither the configuration nor the environment supplies `DSH_HOME`.
pub(super) const MISSING_DSH_HOME_MESSAGE: &str =
    "DeepSeek Harness requires DSH_HOME: set `dsh_home` or the DSH_HOME environment variable";

/// Resolves the mandatory DSH home directory.
///
/// The configuration's `dsh_home` wins, then a non-empty `DSH_HOME` in the configured environment,
/// then a non-empty `DSH_HOME` in the host environment. The runtime never falls back to `~/.dsh`
/// implicitly, so a missing home is reported instead of guessed.
///
/// # Errors
///
/// Returns [`HarnessError::Protocol`] carrying [`MISSING_DSH_HOME_MESSAGE`] when nothing supplies
/// a home directory.
pub(super) fn require_dsh_home(config: &DshHarnessConfig) -> HarnessResult<PathBuf> {
    if let Some(home) = config.dsh_home.as_ref() {
        return Ok(home.clone());
    }
    let configured = config
        .env
        .get(DSH_HOME_ENV)
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from);
    let inherited = std::env::var(DSH_HOME_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from);
    configured
        .or(inherited)
        .ok_or_else(|| HarnessError::protocol(MISSING_DSH_HOME_MESSAGE))
}

/// Builds the runtime argv: `--profile <profile>` and one `--patch <absolute path>` per patch.
///
/// A configured `launch_args_override` replaces the generated vector verbatim (Python passes it to
/// the SDK as `_launch_args`). Patch paths are made absolute against `base` — the configuration's
/// `cwd` or `runtime_cwd` — so the runtime never resolves them against a directory the host did not
/// pin.
///
/// # Errors
///
/// Returns [`Error::Validation`] when a relative patch path cannot be made absolute: no base
/// directory was configured and the process directory is unreadable.
pub(super) fn build_args(config: &DshHarnessConfig, base: Option<&Path>) -> Result<Vec<String>> {
    if let Some(arguments) = config.launch_args_override.as_ref() {
        return Ok(arguments.clone());
    }
    let mut args = Vec::new();
    if let Some(profile) = config.profile.as_deref() {
        args.push("--profile".to_owned());
        args.push(profile.to_owned());
    }
    for patch in &config.patches {
        args.push("--patch".to_owned());
        args.push(absolute_path(patch, base)?.to_string_lossy().into_owned());
    }
    Ok(args)
}

/// Resolves `value` to an absolute path, using `base` and then the process directory.
fn absolute_path(value: &str, base: Option<&Path>) -> Result<PathBuf> {
    let path = Path::new(value);
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    let joined = match base {
        Some(base) => base.join(path),
        None => path.to_path_buf(),
    };
    if joined.is_absolute() {
        return Ok(joined);
    }
    let cwd = std::env::current_dir().map_err(|error| {
        Error::validation(format!(
            "DSH patch path '{value}' is not absolute and cannot be resolved against a working \
             directory: {error}"
        ))
    })?;
    Ok(cwd.join(joined))
}

/// Merges the DSH process environment.
///
/// The child inherits the host environment; this map adds `config.env`, then `context.env`, then
/// the values the adapter owns: `DSH_HOME`, the endpoint pair (`DEEPSEEK_BASE_URL`,
/// `DEEPSEEK_API_KEY`) and, when a custom composition declared one, the system prompt variable.
/// The last writer wins, so an explicitly configured home or endpoint is never overridden by the
/// host environment.
pub(super) fn build_env(
    config: &DshHarnessConfig,
    context: &HarnessContext,
    dsh_home: &Path,
) -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    env.extend(config.env.clone());
    env.extend(context.env.clone());
    env.insert(
        DSH_HOME_ENV.to_owned(),
        dsh_home.to_string_lossy().into_owned(),
    );
    if let Some(base_url) = config.base_url.as_deref() {
        env.insert(DEEPSEEK_BASE_URL_ENV.to_owned(), base_url.to_owned());
    }
    if let Some(api_key) = config.api_key.as_ref() {
        env.insert(
            DEEPSEEK_API_KEY_ENV.to_owned(),
            api_key.expose_secret().to_owned(),
        );
    }
    if let Some(name) = config.system_prompt_env_var.as_deref() {
        if !context.system_prompt.is_empty() {
            env.insert(name.to_owned(), context.system_prompt.clone());
        }
    }
    env
}

/// Returns the working directory of the DSH process.
///
/// The runtime's own directory (`runtime_cwd`) wins, then the host's directory, then the
/// project directory (`cwd`).
pub(super) fn resolve_cwd(config: &DshHarnessConfig, context: &HarnessContext) -> Option<PathBuf> {
    config
        .runtime_cwd
        .clone()
        .or_else(|| context.cwd.as_deref().map(PathBuf::from))
        .or_else(|| config.cwd.clone())
}

/// Returns the project directory skills are installed into: the host's directory, else `cwd`.
pub(super) fn skills_cwd(config: &DshHarnessConfig, context: &HarnessContext) -> PathBuf {
    context
        .cwd
        .as_deref()
        .map(PathBuf::from)
        .or_else(|| config.cwd.clone())
        .unwrap_or_else(|| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    use openjiuwen_common::security::SecretString;
    use openjiuwen_harness_protocol::{HarnessContext, HarnessErrorKind};

    use super::super::DshHarnessConfig;
    use super::{
        absolute_path, build_args, build_env, require_dsh_home, resolve_cwd, skills_cwd,
        DEEPSEEK_API_KEY_ENV, DEEPSEEK_BASE_URL_ENV, DSH_HOME_ENV, MISSING_DSH_HOME_MESSAGE,
    };

    /// Context with the given working directory.
    fn context(cwd: Option<&str>) -> HarnessContext {
        let base = HarnessContext::new("dsh", "agent-1", "host-session", "");
        match cwd {
            Some(cwd) => base.with_cwd(cwd),
            None => base,
        }
    }

    #[test]
    fn configured_home_wins_over_the_environment() {
        let mut config = DshHarnessConfig {
            dsh_home: Some(PathBuf::from("/srv/dsh")),
            ..DshHarnessConfig::default()
        };
        config
            .env
            .insert(DSH_HOME_ENV.to_owned(), "/ignored".to_owned());
        assert_eq!(
            require_dsh_home(&config).expect("the configuration supplies a home"),
            PathBuf::from("/srv/dsh")
        );
    }

    #[test]
    fn configured_environment_supplies_the_home() {
        let mut config = DshHarnessConfig::default();
        config
            .env
            .insert(DSH_HOME_ENV.to_owned(), "/env/dsh".to_owned());
        assert_eq!(
            require_dsh_home(&config).expect("the configured environment supplies a home"),
            PathBuf::from("/env/dsh")
        );
    }

    #[test]
    fn a_blank_configured_home_is_ignored() {
        let mut config = DshHarnessConfig::default();
        config.env.insert(DSH_HOME_ENV.to_owned(), "   ".to_owned());
        match require_dsh_home(&config) {
            Ok(home) => assert_eq!(home, PathBuf::from(std::env::var(DSH_HOME_ENV).unwrap())),
            Err(error) => {
                assert_eq!(error.kind(), HarnessErrorKind::Protocol);
                assert_eq!(error.message(), MISSING_DSH_HOME_MESSAGE);
            }
        }
    }

    #[test]
    fn profile_and_patches_become_arguments() {
        let config = DshHarnessConfig {
            profile: Some("prod".to_owned()),
            patches: vec!["overlay.yml".to_owned(), "/abs/other.yml".to_owned()],
            ..DshHarnessConfig::default()
        };
        let args = build_args(&config, Some(Path::new("/project")))
            .expect("a configured base resolves the relative patch");
        assert_eq!(
            args,
            vec![
                "--profile".to_owned(),
                "prod".to_owned(),
                "--patch".to_owned(),
                "/project/overlay.yml".to_owned(),
                "--patch".to_owned(),
                "/abs/other.yml".to_owned(),
            ]
        );
    }

    #[test]
    fn launch_arguments_override_replaces_the_generated_vector() {
        let config = DshHarnessConfig {
            profile: Some("prod".to_owned()),
            launch_args_override: Some(vec!["--profile".to_owned(), "test".to_owned()]),
            ..DshHarnessConfig::default()
        };
        assert_eq!(
            build_args(&config, None).expect("the override needs no base"),
            vec!["--profile".to_owned(), "test".to_owned()]
        );
    }

    #[test]
    fn a_relative_patch_resolves_against_the_process_directory() {
        let resolved =
            absolute_path("overlay.yml", None).expect("the process directory resolves it");
        assert!(resolved.is_absolute(), "{}", resolved.display());
        assert!(resolved.ends_with("overlay.yml"));
    }

    #[test]
    fn the_environment_carries_the_home_and_the_endpoint() {
        let mut config = DshHarnessConfig {
            base_url: Some("https://api.example.test".to_owned()),
            api_key: Some(SecretString::new("sk-test")),
            system_prompt_env_var: Some("DSH_SYSTEM_PROMPT".to_owned()),
            ..DshHarnessConfig::default()
        };
        config
            .env
            .insert("DEEPSEEK_DEBUG".to_owned(), "1".to_owned());
        let mut host = context(None);
        host.env = BTreeMap::from([("HOST_FLAG".to_owned(), "yes".to_owned())]);
        host.system_prompt = "be brief".to_owned();
        let env = build_env(&config, &host, Path::new("/home/dsh"));
        assert_eq!(env.get("DEEPSEEK_DEBUG").map(String::as_str), Some("1"));
        assert_eq!(env.get("HOST_FLAG").map(String::as_str), Some("yes"));
        assert_eq!(env.get(DSH_HOME_ENV).map(String::as_str), Some("/home/dsh"));
        assert_eq!(
            env.get(DEEPSEEK_BASE_URL_ENV).map(String::as_str),
            Some("https://api.example.test")
        );
        assert_eq!(
            env.get(DEEPSEEK_API_KEY_ENV).map(String::as_str),
            Some("sk-test")
        );
        assert_eq!(
            env.get("DSH_SYSTEM_PROMPT").map(String::as_str),
            Some("be brief")
        );
    }

    #[test]
    fn the_host_environment_never_overrides_the_home() {
        let config = DshHarnessConfig::default();
        let mut host = context(None);
        host.env = BTreeMap::from([(DSH_HOME_ENV.to_owned(), "/host/dsh".to_owned())]);
        let env = build_env(&config, &host, Path::new("/configured/dsh"));
        assert_eq!(
            env.get(DSH_HOME_ENV).map(String::as_str),
            Some("/configured/dsh")
        );
    }

    #[test]
    fn the_runtime_directory_wins_the_working_directory() {
        let config = DshHarnessConfig {
            cwd: Some(PathBuf::from("/project")),
            ..DshHarnessConfig::default()
        };
        assert_eq!(
            resolve_cwd(&config, &context(Some("/host"))),
            Some(PathBuf::from("/host"))
        );
        assert_eq!(
            resolve_cwd(&config, &context(None)),
            Some(PathBuf::from("/project"))
        );
        let config = DshHarnessConfig {
            cwd: Some(PathBuf::from("/project")),
            runtime_cwd: Some(PathBuf::from("/runtime")),
            ..DshHarnessConfig::default()
        };
        assert_eq!(
            resolve_cwd(&config, &context(Some("/host"))),
            Some(PathBuf::from("/runtime"))
        );
        let config = DshHarnessConfig::default();
        assert_eq!(
            resolve_cwd(&config, &context(Some("/host"))),
            Some(PathBuf::from("/host"))
        );
        assert_eq!(resolve_cwd(&config, &context(None)), None);
    }

    #[test]
    fn skills_install_into_the_project_directory() {
        let config = DshHarnessConfig {
            cwd: Some(PathBuf::from("/project")),
            ..DshHarnessConfig::default()
        };
        assert_eq!(
            skills_cwd(&config, &context(Some("/host"))),
            PathBuf::from("/host")
        );
        assert_eq!(
            skills_cwd(&config, &context(None)),
            PathBuf::from("/project")
        );
        assert_eq!(
            skills_cwd(&DshHarnessConfig::default(), &context(None)),
            PathBuf::from(".")
        );
    }
}
