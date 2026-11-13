//! Resolution of every file path and environment switch the CLI owns.
//!
//! All CLI file locations are resolved here so they stay injectable and testable.
//! Every function is pure: it reads the injected [`EnvSource`] and returns a
//! `PathBuf` or `bool`, and nothing touches the disk.

use std::path::{Path, PathBuf};

use openjiuwen_common::config::OpenJiuwenHome;

use super::config::EnvSource;

/// Name of the persistent REPL history file under the CLI home directory.
pub const HISTORY_FILENAME: &str = ".openjiuwen_history";

/// Name of the permanent allow-rule file under the CLI home directory.
pub const ALLOW_RULES_FILENAME: &str = "permissions.yaml";

/// Name of the MCP server configuration file.
pub const MCP_CONFIG_FILENAME: &str = "mcp.json";

/// Name of the per-project directory that may hold an MCP configuration file.
pub const MCP_CONFIG_DIRNAME: &str = ".openjiuwen";

/// Name of the settings file under the CLI home directory.
pub const SETTINGS_FILENAME: &str = "settings.json";

/// Maximum number of history entries retained in the history file.
pub const MAX_HISTORY_ENTRIES: usize = 1000;

/// Disables colour when set to any non-blank value: `NO_COLOR`.
pub const NO_COLOR: &str = "NO_COLOR";

/// Overrides the history file location: `OPENJIUWEN_HISTORY`.
pub const OPENJIUWEN_HISTORY: &str = "OPENJIUWEN_HISTORY";

/// Enables replaying the history file at startup: `OPENJIUWEN_HISTORY_REPLAY`.
pub const OPENJIUWEN_HISTORY_REPLAY: &str = "OPENJIUWEN_HISTORY_REPLAY";

/// Overrides the MCP configuration file location: `OPENJIUWEN_MCP_CONFIG`.
pub const OPENJIUWEN_MCP_CONFIG: &str = "OPENJIUWEN_MCP_CONFIG";

/// Skips the first-run onboarding wizard when set: `OPENJIUWEN_NO_ONBOARDING`.
pub const OPENJIUWEN_NO_ONBOARDING: &str = "OPENJIUWEN_NO_ONBOARDING";

/// Returns the settings file location, `<home>/settings.json`.
#[must_use]
pub fn settings_path(home: &OpenJiuwenHome) -> PathBuf {
    home.join(SETTINGS_FILENAME)
}

/// Returns the REPL history file location.
///
/// `OPENJIUWEN_HISTORY` wins over the default `<home>/.openjiuwen_history`, with
/// a relative value resolved against the home root and an absolute one used
/// as-is.
#[must_use]
pub fn history_path(home: &OpenJiuwenHome, env: &dyn EnvSource) -> PathBuf {
    match env.get(OPENJIUWEN_HISTORY) {
        Some(value) => anchored(home, &value),
        None => home.join(HISTORY_FILENAME),
    }
}

/// Returns the permanent allow-rule file location, `<home>/permissions.yaml`.
#[must_use]
pub fn allow_rules_path(home: &OpenJiuwenHome) -> PathBuf {
    home.join(ALLOW_RULES_FILENAME)
}

/// Returns the MCP configuration files to load, in precedence order.
///
/// `OPENJIUWEN_MCP_CONFIG` pins the list to exactly one file, with a relative
/// value resolved against the home root. Otherwise the home-level `mcp.json`
/// comes first and the workspace-level `.openjiuwen/mcp.json` second; later
/// entries override earlier ones by server name.
#[must_use]
pub fn mcp_config_paths(
    home: &OpenJiuwenHome,
    workspace: &Path,
    env: &dyn EnvSource,
) -> Vec<PathBuf> {
    if let Some(value) = env.get(OPENJIUWEN_MCP_CONFIG) {
        return vec![anchored(home, &value)];
    }
    vec![
        home.join(MCP_CONFIG_FILENAME),
        workspace.join(MCP_CONFIG_DIRNAME).join(MCP_CONFIG_FILENAME),
    ]
}

/// Returns whether ANSI colour should be emitted.
///
/// Colour needs a terminal, no `--no-color` flag and no `NO_COLOR` variable.
#[must_use]
pub fn color_enabled(no_color_flag: bool, env: &dyn EnvSource, is_terminal: bool) -> bool {
    is_terminal && !no_color_flag && env.get(NO_COLOR).is_none()
}

/// Returns whether the first-run onboarding wizard is suppressed.
#[must_use]
pub fn onboarding_disabled(env: &dyn EnvSource) -> bool {
    env.get(OPENJIUWEN_NO_ONBOARDING).is_some()
}

/// Returns whether the REPL replays the stored history at startup.
#[must_use]
pub fn history_replay_enabled(env: &dyn EnvSource) -> bool {
    env.get(OPENJIUWEN_HISTORY_REPLAY).is_some()
}

/// Resolves `value` against the home root when it is a relative path.
fn anchored(home: &OpenJiuwenHome, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        home.root().join(path)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::cli::config::MapEnv;

    /// Returns the home used by the path assertions.
    fn home() -> OpenJiuwenHome {
        OpenJiuwenHome::from_root("cli-home")
    }

    /// Returns an absolute path usable as an environment override.
    fn absolute_override(name: &str) -> PathBuf {
        std::env::temp_dir().join(name)
    }

    #[test]
    fn default_history_path_lives_under_the_home_root() {
        let env = MapEnv::new();
        assert_eq!(
            history_path(&home(), &env),
            PathBuf::from("cli-home").join(".openjiuwen_history")
        );
    }

    #[test]
    fn absolute_history_override_is_used_as_is() {
        let target = absolute_override("custom-history");
        let env = MapEnv::from_pairs([(OPENJIUWEN_HISTORY, target.to_str().unwrap())]);
        assert_eq!(history_path(&home(), &env), target);
    }

    #[test]
    fn relative_history_override_is_resolved_against_the_home_root() {
        let env = MapEnv::from_pairs([(OPENJIUWEN_HISTORY, "state/history.txt")]);
        assert_eq!(
            history_path(&home(), &env),
            PathBuf::from("cli-home").join("state/history.txt")
        );
    }

    #[test]
    fn blank_history_override_falls_back_to_the_default() {
        let env = MapEnv::from_pairs([(OPENJIUWEN_HISTORY, "   ")]);
        assert_eq!(
            history_path(&home(), &env),
            PathBuf::from("cli-home").join(HISTORY_FILENAME)
        );
    }

    #[test]
    fn settings_and_allow_rules_live_under_the_home_root() {
        assert_eq!(
            settings_path(&home()),
            PathBuf::from("cli-home").join("settings.json")
        );
        assert_eq!(
            allow_rules_path(&home()),
            PathBuf::from("cli-home").join("permissions.yaml")
        );
    }

    #[test]
    fn mcp_config_defaults_to_home_then_workspace() {
        let env = MapEnv::new();
        let workspace = Path::new("/project");
        let paths = mcp_config_paths(&home(), workspace, &env);
        assert_eq!(
            paths,
            vec![
                PathBuf::from("cli-home").join("mcp.json"),
                Path::new("/project").join(".openjiuwen").join("mcp.json"),
            ]
        );
    }

    #[test]
    fn absolute_mcp_config_override_replaces_the_default_list() {
        let target = absolute_override("custom-mcp.json");
        let env = MapEnv::from_pairs([(OPENJIUWEN_MCP_CONFIG, target.to_str().unwrap())]);
        assert_eq!(
            mcp_config_paths(&home(), Path::new("/project"), &env),
            vec![target]
        );
    }

    #[test]
    fn relative_mcp_config_override_is_resolved_against_the_home_root() {
        let env = MapEnv::from_pairs([(OPENJIUWEN_MCP_CONFIG, "conf/mcp.json")]);
        assert_eq!(
            mcp_config_paths(&home(), Path::new("/project"), &env),
            vec![PathBuf::from("cli-home").join("conf/mcp.json")]
        );
    }

    #[test]
    fn color_enabled_truth_table() {
        let without = MapEnv::new();
        let with = MapEnv::from_pairs([(NO_COLOR, "1")]);
        for (terminal, flag, no_color, expected) in [
            (true, false, false, true),
            (true, false, true, false),
            (true, true, false, false),
            (true, true, true, false),
            (false, false, false, false),
            (false, false, true, false),
            (false, true, false, false),
            (false, true, true, false),
        ] {
            let env = if no_color { &with } else { &without };
            assert_eq!(
                color_enabled(flag, env, terminal),
                expected,
                "terminal={terminal} flag={flag} no_color={no_color}"
            );
        }
    }

    #[test]
    fn onboarding_switch_is_set_or_unset() {
        assert!(!onboarding_disabled(&MapEnv::new()));
        assert!(onboarding_disabled(&MapEnv::from_pairs([(
            OPENJIUWEN_NO_ONBOARDING,
            "1"
        )])));
        assert!(!onboarding_disabled(&MapEnv::from_pairs([(
            OPENJIUWEN_NO_ONBOARDING,
            ""
        )])));
    }

    #[test]
    fn history_replay_switch_is_set_or_unset() {
        assert!(!history_replay_enabled(&MapEnv::new()));
        assert!(history_replay_enabled(&MapEnv::from_pairs([(
            OPENJIUWEN_HISTORY_REPLAY,
            "1"
        )])));
        assert!(!history_replay_enabled(&MapEnv::from_pairs([(
            OPENJIUWEN_HISTORY_REPLAY,
            "   "
        )])));
    }
}
