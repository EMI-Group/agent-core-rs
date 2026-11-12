//! Reader and writer for the skill library's on/off state file.
//!
//! Ports `openjiuwen/harness/skills/library_state.py`. A library keeps a library-wide
//! kill switch in `skills_state.json` next to the skills themselves:
//!
//! ```json
//! {"skill_configs": {"gamma": {"enabled": false}}}
//! ```
//!
//! A skill switched off there is unavailable to every agent that reads the library.
//! The format is read defensively: a missing, unreadable or malformed file means
//! "nothing is switched off", because a corrupted file must not silently blank out an
//! agent's whole skill view.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};
use serde_json::{json, Map, Value};

/// Basename of the library-wide skill on/off state file.
pub const SKILLS_STATE_FILENAME: &str = "skills_state.json";

/// Parses state-file text into the names switched off.
///
/// Only a skill whose stored config is an object with an exact JSON boolean
/// `"enabled": false` is disabled; `0`, `"false"`, a missing key and non-object entries
/// do not disable anything. Text that is not a JSON object, or whose `skill_configs`
/// value is not an object, disables nothing.
#[must_use]
pub fn parse_disabled_skills(text: &str) -> BTreeSet<String> {
    serde_json::from_str::<Value>(text)
        .map_or_else(|_| BTreeSet::new(), |value| disabled_from_value(&value))
}

/// Reads one library root's disabled skill names, failing open.
///
/// Only `<root>/skills_state.json` is read, and only when it is a regular file (symlinks
/// are followed). A missing file is silent; an unreadable or malformed file emits one
/// warning and yields an empty set so a corrupt state file never hides every skill.
#[must_use]
pub async fn read_disabled_skills(root: &Path) -> BTreeSet<String> {
    let path = root.join(SKILLS_STATE_FILENAME);
    let Ok(metadata) = tokio::fs::metadata(&path).await else {
        return BTreeSet::new();
    };
    if !metadata.is_file() {
        return BTreeSet::new();
    }
    let Ok(text) = tokio::fs::read_to_string(&path).await else {
        tracing::warn!(
            path = %path.display(),
            "[SkillLibraryState] failed to read state file; treating it as no Skill disabled"
        );
        return BTreeSet::new();
    };
    match serde_json::from_str::<Value>(&text) {
        Ok(value) => disabled_from_value(&value),
        Err(error) => {
            tracing::warn!(
                path = %path.display(),
                error = %error,
                "[SkillLibraryState] failed to parse state file; treating it as no Skill disabled"
            );
            BTreeSet::new()
        }
    }
}

/// Collects the disabled skill names of every given library root.
///
/// Roots that hold no state file contribute nothing; duplicates are merged and the
/// result is ordered because the accumulator is a [`BTreeSet`].
#[must_use]
pub async fn collect_disabled_skills(roots: &[PathBuf]) -> BTreeSet<String> {
    let mut disabled = BTreeSet::new();
    for root in roots {
        disabled.extend(read_disabled_skills(root).await);
    }
    disabled
}

/// Writes the disabled skill names into one library root's state file.
///
/// The root is created when missing. Only disabled names are written, each as
/// `{"enabled": false}`, in a pretty-printed object with a trailing newline. Other
/// entries in an existing file are not preserved (the library owns the file).
///
/// # Errors
///
/// Returns [`Error::sys_op`] when the root cannot be created, the state cannot be
/// serialized, or the file cannot be written.
pub async fn save_disabled_skills(root: &Path, disabled: &BTreeSet<String>) -> Result<()> {
    tokio::fs::create_dir_all(root)
        .await
        .map_err(|error| Error::sys_op(format!("failed to create {}: {error}", root.display())))?;
    let configs: Map<String, Value> = disabled
        .iter()
        .map(|name| (name.clone(), json!({ "enabled": false })))
        .collect();
    let state = json!({ "skill_configs": Value::Object(configs) });
    let path = root.join(SKILLS_STATE_FILENAME);
    let text = serde_json::to_string_pretty(&state).map_err(|error| {
        Error::sys_op(format!("failed to serialize {}: {error}", path.display()))
    })?;
    tokio::fs::write(&path, format!("{text}\n"))
        .await
        .map_err(|error| Error::sys_op(format!("failed to write {}: {error}", path.display())))
}

/// Extracts disabled names from an already-parsed JSON value.
fn disabled_from_value(value: &Value) -> BTreeSet<String> {
    let mut disabled = BTreeSet::new();
    let Some(configs) = value.get("skill_configs").and_then(Value::as_object) else {
        return disabled;
    };
    for (name, config) in configs {
        if config.get("enabled").and_then(Value::as_bool) == Some(false) {
            disabled.insert(name.clone());
        }
    }
    disabled
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn temp_dir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    fn state_text(names: &[&str]) -> String {
        let configs: Map<String, Value> = names
            .iter()
            .map(|name| ((*name).to_string(), json!({ "enabled": false })))
            .collect();
        serde_json::to_string(&json!({ "skill_configs": Value::Object(configs) })).unwrap()
    }

    fn write_state(root: &Path, text: &str) {
        std::fs::create_dir_all(root).unwrap();
        std::fs::write(root.join(SKILLS_STATE_FILENAME), text).unwrap();
    }

    #[test]
    fn only_an_exact_false_boolean_disables_a_skill() {
        let text = r#"{"skill_configs": {
            "alpha": {"enabled": true},
            "beta": {},
            "gamma": {"enabled": false}
        }}"#;
        let disabled = parse_disabled_skills(text);
        assert_eq!(disabled.into_iter().collect::<Vec<_>>(), vec!["gamma"]);
    }

    #[test]
    fn non_boolean_enabled_values_do_not_disable() {
        let text = r#"{"skill_configs": {
            "alpha": {"enabled": 0},
            "beta": {"enabled": "false"},
            "gamma": {"enabled": null}
        }}"#;
        assert_eq!(parse_disabled_skills(text), BTreeSet::new());
    }

    #[test]
    fn malformed_state_text_disables_nothing() {
        assert_eq!(parse_disabled_skills("{not json"), BTreeSet::new());
        assert_eq!(parse_disabled_skills(r#"["gamma"]"#), BTreeSet::new());
        assert_eq!(
            parse_disabled_skills(r#"{"skill_configs": ["gamma"]}"#),
            BTreeSet::new()
        );
        assert_eq!(
            parse_disabled_skills(r#"{"skill_configs": {"gamma": "disabled"}}"#),
            BTreeSet::new()
        );
    }

    #[tokio::test]
    async fn missing_directory_and_file_are_silently_empty() {
        let temp = temp_dir();
        assert_eq!(
            read_disabled_skills(&temp.path().join("missing")).await,
            BTreeSet::new()
        );
        std::fs::create_dir_all(temp.path()).unwrap();
        assert_eq!(read_disabled_skills(temp.path()).await, BTreeSet::new());
    }

    #[tokio::test]
    async fn unreadable_state_is_empty_without_error() {
        let temp = temp_dir();
        write_state(temp.path(), "{not json");
        assert_eq!(read_disabled_skills(temp.path()).await, BTreeSet::new());
    }

    #[tokio::test]
    async fn collect_merges_roots_sorted_and_deduped() {
        let first = temp_dir();
        let second = temp_dir();
        write_state(first.path(), &state_text(&["zeta"]));
        write_state(second.path(), &state_text(&["zeta", "alpha"]));

        let roots = vec![first.path().to_path_buf(), second.path().to_path_buf()];
        let disabled = collect_disabled_skills(&roots).await;
        assert_eq!(
            disabled.into_iter().collect::<Vec<_>>(),
            vec!["alpha", "zeta"]
        );
    }

    #[tokio::test]
    async fn save_and_collect_round_trip() {
        let temp = temp_dir();
        let root = temp.path().join("nested").join("library");
        let disabled = BTreeSet::from(["alpha".to_string(), "gamma".to_string()]);

        save_disabled_skills(&root, &disabled).await.unwrap();

        let text = std::fs::read_to_string(root.join(SKILLS_STATE_FILENAME)).unwrap();
        assert!(text.contains("skill_configs"));
        assert!(text.contains("\"alpha\""));
        assert!(text.contains("\"gamma\""));
        assert!(text.contains("\"enabled\": false"));
        assert!(text.ends_with('\n'));

        let collected = collect_disabled_skills(&[root]).await;
        assert_eq!(collected, disabled);
    }

    #[tokio::test]
    async fn save_writes_no_enabled_skills() {
        let temp = temp_dir();
        let disabled = BTreeSet::new();
        save_disabled_skills(temp.path(), &disabled).await.unwrap();
        let text = std::fs::read_to_string(temp.path().join(SKILLS_STATE_FILENAME)).unwrap();
        assert!(text.contains("\"skill_configs\": {}"));
        assert_eq!(
            collect_disabled_skills(&[temp.path().to_path_buf()]).await,
            BTreeSet::new()
        );
    }
}
