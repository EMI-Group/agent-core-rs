//! Team-memory configuration read from [`TeamAgentSpec`] metadata.
//!
//! Python parity source: `openjiuwen/agent_teams/memory/` (v0.1.20), where the team memory
//! manager is configured by the team spec rather than by a typed field. Rust keeps that
//! convention: everything the host tunes lives under the optional
//! [`TeamAgentSpec::metadata`](crate::schema::TeamAgentSpec::metadata) key [`MEMORY_METADATA_KEY`].
//!
//! Parsing is deliberately tolerant, exactly like
//! [`checkpoint_config`](crate::runtime::spec_meta::checkpoint_config): a missing, `null` or
//! blank value keeps the default configuration, a malformed value logs one warning and keeps
//! the default for that key only, and an unknown store kind falls back to the in-memory store.
//! A runtime must never fail because an optional knob is malformed.
//!
//! Accepted shapes:
//!
//! ```json
//! {
//!   "enabled": true,
//!   "store": "memory" | "file" | "/var/lib/team-store" | {"kind": "file", "dir": "/var/lib"},
//!   "extractor": "rule" | "llm",
//!   "export_on_stop": true,
//!   "import_on_build": true,
//!   "hooks": {"member_profile_on_join": true, "handoff_importance": 0.6}
//! }
//! ```
//!
//! The `hooks` block is deserialized straight into
//! [`TeamMemoryHookConfig`] from `openjiuwen_memory::team`, so every switch and importance the
//! engine understands is reachable from metadata without a second schema here.
use std::fmt;
use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_memory::team::TeamMemoryHookConfig;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use tracing::warn;

use crate::schema::TeamAgentSpec;

/// Metadata key holding the team-memory configuration.
pub const MEMORY_METADATA_KEY: &str = "memory";

/// The store name that selects the in-memory backend.
const STORE_MEMORY: &str = "memory";
/// The store name that selects the durable document backend.
const STORE_FILE: &str = "file";
/// The object key naming the store backend.
const KIND_KEY: &str = "kind";
/// The object key naming the store directory.
const DIR_KEY: &str = "dir";

/// Where a team keeps its memories.
///
/// The default, [`MemoryStoreConfig::Memory`], is the process-local
/// [`InMemoryTeamMemoryStore`](openjiuwen_memory::team::InMemoryTeamMemoryStore) and persists
/// nothing between runs. [`MemoryStoreConfig::File`] names the directory whose
/// `TEAM_MEMORY.md` document is the durable image of the store: this crate does not enable the
/// `openjiuwen-memory` `sqlite` feature, so a file-backed wiring keeps the in-memory facade and
/// persists through the exported document instead (see
/// [`TeamMemoryWiring`](crate::memory::wiring::TeamMemoryWiring)).
///
/// Deserialization is as tolerant as the metadata parser: it accepts a bare name
/// (`"memory"`, `"file"`), a bare directory path, or a `{"kind": …, "dir": …}` object, and an
/// unrecognized value resolves to [`MemoryStoreConfig::Memory`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryStoreConfig {
    /// The process-local in-memory store.
    Memory,
    /// A durable store rooted at `dir`; `None` keeps the workspace root.
    File {
        /// Directory holding the `TEAM_MEMORY.md` document.
        dir: Option<PathBuf>,
    },
}

impl Default for MemoryStoreConfig {
    fn default() -> Self {
        Self::Memory
    }
}

impl MemoryStoreConfig {
    /// Returns `true` when the durable file backend is selected.
    #[must_use]
    pub fn is_file(&self) -> bool {
        matches!(self, Self::File { .. })
    }

    /// The configured store directory, when this is a file-backed configuration.
    #[must_use]
    pub fn dir(&self) -> Option<&Path> {
        match self {
            Self::Memory => None,
            Self::File { dir } => dir.as_deref(),
        }
    }

    /// Resolves a bare store name or directory path.
    fn from_name(name: &str) -> Self {
        let name = name.trim();
        if name.is_empty() || name.eq_ignore_ascii_case(STORE_MEMORY) {
            Self::Memory
        } else if name.eq_ignore_ascii_case(STORE_FILE) {
            Self::File { dir: None }
        } else {
            Self::File {
                dir: Some(PathBuf::from(name)),
            }
        }
    }

    /// Resolves one JSON `store` value, reporting why it degraded when it did.
    fn from_json(value: &Value) -> (Self, Option<&'static str>) {
        match value {
            Value::Null => (Self::Memory, None),
            Value::String(name) => (Self::from_name(name), None),
            Value::Object(map) => {
                let dir = dir_from(map.get(DIR_KEY));
                let dir_problem = map
                    .get(DIR_KEY)
                    .is_some_and(|value| !matches!(value, Value::Null | Value::String(_)));
                let kind_problem = map
                    .get(KIND_KEY)
                    .is_some_and(|value| !matches!(value, Value::Null | Value::String(_)));
                let dir_present = dir.is_some();
                let kind = map
                    .get(KIND_KEY)
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|name| !name.is_empty());
                let config = match kind {
                    None => dir.map_or(Self::Memory, |dir| Self::File { dir: Some(dir) }),
                    Some(name) if name.eq_ignore_ascii_case(STORE_FILE) => Self::File { dir },
                    Some(name) if name.eq_ignore_ascii_case(STORE_MEMORY) => Self::Memory,
                    Some(_) => Self::Memory,
                };
                let problem = if kind_problem {
                    Some("the store `kind` is not a string")
                } else if dir_problem {
                    Some("the store `dir` is not a path string")
                } else if kind.is_some_and(|name| {
                    !name.eq_ignore_ascii_case(STORE_FILE)
                        && !name.eq_ignore_ascii_case(STORE_MEMORY)
                }) {
                    Some("the store kind is neither `memory` nor `file`")
                } else if matches!(config, Self::Memory) && dir_present {
                    Some("ignoring the store dir: the in-memory store persists nothing")
                } else {
                    None
                };
                (config, problem)
            }
            _ => (
                Self::Memory,
                Some("the store is not a name, a path, or an object"),
            ),
        }
    }
}

/// Reads an optional store directory: a blank string, `null` or a non-string is absent.
fn dir_from(value: Option<&Value>) -> Option<PathBuf> {
    match value {
        None | Some(Value::Null) => None,
        Some(Value::String(path)) if path.trim().is_empty() => None,
        Some(Value::String(path)) => Some(PathBuf::from(path.trim())),
        Some(_) => None,
    }
}

impl Serialize for MemoryStoreConfig {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::Memory => serializer.serialize_str(STORE_MEMORY),
            Self::File { dir: None } => {
                use serde::ser::SerializeMap as _;
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry(KIND_KEY, STORE_FILE)?;
                map.end()
            }
            Self::File { dir: Some(dir) } => {
                use serde::ser::SerializeMap as _;
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry(KIND_KEY, STORE_FILE)?;
                map.serialize_entry(DIR_KEY, &dir.to_string_lossy())?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for MemoryStoreConfig {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        Ok(Self::from_json(&value).0)
    }
}

/// Which extractor turns a buffered session into team memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryExtractorKind {
    /// The deterministic rule extractor (the default; no model call).
    Rule,
    /// The model-backed extractor, which degrades to the rules on failure.
    Llm,
}

impl Default for MemoryExtractorKind {
    fn default() -> Self {
        Self::Rule
    }
}

impl fmt::Display for MemoryExtractorKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Rule => "rule",
            Self::Llm => "llm",
        })
    }
}

/// The team-memory configuration of one team runtime.
///
/// [`memory_config`] parses it from [`TeamAgentSpec`] metadata; the defaults are the fully
/// working configuration (memory on, rule extractor, document exported on stop and imported on
/// build).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemoryConfig {
    /// Master switch: when `false`, no hook, export or bundle is built at all.
    pub enabled: bool,
    /// Where the memories live between runs.
    pub store: MemoryStoreConfig,
    /// Which extractor the session-end flush runs.
    pub extractor: MemoryExtractorKind,
    /// Write `TEAM_MEMORY.md` when the team stops.
    pub export_on_stop: bool,
    /// Import an existing `TEAM_MEMORY.md` when the team is built.
    pub import_on_build: bool,
    /// The lifecycle-hook switches and importances handed to the engine.
    pub hooks: TeamMemoryHookConfig,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            store: MemoryStoreConfig::default(),
            extractor: MemoryExtractorKind::default(),
            export_on_stop: true,
            import_on_build: true,
            hooks: TeamMemoryHookConfig::default(),
        }
    }
}

impl MemoryConfig {
    /// Validates the hook switches and the store directory.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when an importance is out of range, `max_pending_turns` is zero,
    /// or a file store names a blank directory.
    pub fn validate(&self) -> Result<()> {
        self.hooks.validate()?;
        if let MemoryStoreConfig::File { dir: Some(dir) } = &self.store {
            if dir.as_os_str().to_string_lossy().trim().is_empty() {
                return Err(Error::validation(
                    "team memory store `dir` must not be blank".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

/// Reads the team-memory configuration out of `spec`'s metadata.
///
/// A missing, `null` or blank `memory` value keeps [`MemoryConfig::default`]; a malformed value
/// logs one warning and keeps the default for the key it could not read, so a typo never
/// disables team memory by accident.
#[must_use]
pub fn memory_config(spec: &TeamAgentSpec) -> MemoryConfig {
    let Some(value) = spec.metadata.get(MEMORY_METADATA_KEY) else {
        return MemoryConfig::default();
    };
    match value {
        Value::Null => MemoryConfig::default(),
        Value::String(text) if text.trim().is_empty() => MemoryConfig::default(),
        Value::Object(map) => parse_block(spec, map),
        _ => {
            warn!(
                team = %spec.team_name,
                "ignoring memory metadata that is not an object; team memory stays at its \
                 default configuration"
            );
            MemoryConfig::default()
        }
    }
}

/// Parses one `memory` object key by key, degrading every malformed key on its own.
fn parse_block(spec: &TeamAgentSpec, map: &serde_json::Map<String, Value>) -> MemoryConfig {
    let default = MemoryConfig::default();
    MemoryConfig {
        enabled: parse_bool(spec, map, "enabled", default.enabled),
        store: parse_store(spec, map.get("store")),
        extractor: parse_extractor(spec, map.get("extractor")),
        export_on_stop: parse_bool(spec, map, "export_on_stop", default.export_on_stop),
        import_on_build: parse_bool(spec, map, "import_on_build", default.import_on_build),
        hooks: parse_hooks(spec, map.get("hooks")),
    }
}

/// Reads one boolean key; a non-boolean warns and keeps `fallback`.
fn parse_bool(
    spec: &TeamAgentSpec,
    map: &serde_json::Map<String, Value>,
    key: &str,
    fallback: bool,
) -> bool {
    match map.get(key) {
        None | Some(Value::Null) => fallback,
        Some(Value::Bool(value)) => *value,
        Some(_) => {
            warn!(
                team = %spec.team_name,
                key,
                "ignoring a team memory switch that is not a boolean; keeping its default"
            );
            fallback
        }
    }
}

/// Reads the `store` key: a name, a directory path, or a `{kind, dir}` object.
fn parse_store(spec: &TeamAgentSpec, value: Option<&Value>) -> MemoryStoreConfig {
    match value {
        None | Some(Value::Null) => MemoryStoreConfig::Memory,
        Some(value) => {
            let (store, problem) = MemoryStoreConfig::from_json(value);
            if let Some(problem) = problem {
                warn!(team = %spec.team_name, problem, "degrading the team memory store");
            }
            store
        }
    }
}

/// Reads the `extractor` key; an unknown name warns and keeps the rule extractor.
fn parse_extractor(spec: &TeamAgentSpec, value: Option<&Value>) -> MemoryExtractorKind {
    let fallback = MemoryExtractorKind::default();
    match value {
        None | Some(Value::Null) => fallback,
        Some(Value::String(name)) => match name.trim().to_ascii_lowercase().as_str() {
            "rule" => MemoryExtractorKind::Rule,
            "llm" => MemoryExtractorKind::Llm,
            _ => {
                warn!(
                    team = %spec.team_name,
                    extractor = name,
                    "unknown team memory extractor; keeping the rule extractor"
                );
                fallback
            }
        },
        Some(_) => {
            warn!(
                team = %spec.team_name,
                "ignoring a team memory extractor that is not a name; keeping the rule extractor"
            );
            fallback
        }
    }
}

/// Reads the `hooks` key through serde; a malformed block warns and keeps the defaults.
fn parse_hooks(spec: &TeamAgentSpec, value: Option<&Value>) -> TeamMemoryHookConfig {
    match value {
        None | Some(Value::Null) => TeamMemoryHookConfig::default(),
        Some(value) => match serde_json::from_value::<TeamMemoryHookConfig>(value.clone()) {
            Ok(hooks) => hooks,
            Err(error) => {
                warn!(
                    team = %spec.team_name,
                    %error,
                    "ignoring malformed team memory hooks; keeping the default hooks"
                );
                TeamMemoryHookConfig::default()
            }
        },
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;

    use super::*;

    /// Builds a spec whose metadata carries one `memory` value.
    fn spec_with(value: Value) -> TeamAgentSpec {
        let mut spec = TeamAgentSpec {
            team_name: "team-1".to_owned(),
            ..TeamAgentSpec::default()
        };
        spec.metadata = BTreeMap::from([(MEMORY_METADATA_KEY.to_owned(), value)]);
        spec
    }

    #[test]
    fn absent_or_blank_metadata_keeps_the_defaults() {
        let absent = TeamAgentSpec {
            team_name: "team-1".to_owned(),
            ..TeamAgentSpec::default()
        };
        assert_eq!(memory_config(&absent), MemoryConfig::default());
        for value in [Value::Null, json!(""), json!("   ")] {
            assert_eq!(memory_config(&spec_with(value)), MemoryConfig::default());
        }
    }

    #[test]
    fn the_defaults_are_the_working_configuration() {
        let config = MemoryConfig::default();
        assert!(config.enabled);
        assert_eq!(config.store, MemoryStoreConfig::Memory);
        assert_eq!(config.extractor, MemoryExtractorKind::Rule);
        assert!(config.export_on_stop);
        assert!(config.import_on_build);
        assert_eq!(config.hooks, TeamMemoryHookConfig::default());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn a_bare_object_is_read_key_by_key() {
        let config = memory_config(&spec_with(json!({
            "enabled": false,
            "store": "memory",
            "extractor": "llm",
            "export_on_stop": false,
            "import_on_build": false,
            "hooks": {"member_profile_on_join": false, "handoff_importance": 0.9},
        })));
        assert!(!config.enabled);
        assert_eq!(config.store, MemoryStoreConfig::Memory);
        assert_eq!(config.extractor, MemoryExtractorKind::Llm);
        assert!(!config.export_on_stop);
        assert!(!config.import_on_build);
        assert!(!config.hooks.member_profile_on_join);
        assert!((config.hooks.handoff_importance - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn every_store_shape_resolves() {
        let cases = [
            (json!("memory"), MemoryStoreConfig::Memory),
            (json!("file"), MemoryStoreConfig::File { dir: None }),
            (
                json!("/var/lib/team-store"),
                MemoryStoreConfig::File {
                    dir: Some(PathBuf::from("/var/lib/team-store")),
                },
            ),
            (
                json!({"kind": "file", "dir": "/var/lib/team-store"}),
                MemoryStoreConfig::File {
                    dir: Some(PathBuf::from("/var/lib/team-store")),
                },
            ),
            (
                json!({"kind": "file"}),
                MemoryStoreConfig::File { dir: None },
            ),
            // `dir` without `kind` still selects the file store.
            (
                json!({"dir": "/var/lib/team-store"}),
                MemoryStoreConfig::File {
                    dir: Some(PathBuf::from("/var/lib/team-store")),
                },
            ),
            // An unknown kind falls back to memory, as does a non-string value.
            (json!({"kind": "postgresql"}), MemoryStoreConfig::Memory),
            (json!(42), MemoryStoreConfig::Memory),
            (json!([]), MemoryStoreConfig::Memory),
        ];
        for (value, expected) in cases {
            let config = memory_config(&spec_with(json!({"store": value})));
            assert_eq!(config.store, expected, "for store {value}");
        }
    }

    #[test]
    fn malformed_keys_degrade_one_by_one() {
        let config = memory_config(&spec_with(json!({
            "enabled": "yes",
            "store": {"kind": 3},
            "extractor": "bogus",
            "export_on_stop": 1,
            "hooks": "nonsense",
        })));
        // Every malformed switch keeps its default; one typo never disables the rest.
        assert!(config.enabled);
        assert_eq!(config.store, MemoryStoreConfig::Memory);
        assert_eq!(config.extractor, MemoryExtractorKind::Rule);
        assert!(config.export_on_stop);
        assert_eq!(config.hooks, TeamMemoryHookConfig::default());
    }

    #[test]
    fn a_non_object_value_keeps_the_defaults() {
        for value in [json!(true), json!(7), json!(["memory"])] {
            assert_eq!(memory_config(&spec_with(value)), MemoryConfig::default());
        }
    }

    #[test]
    fn an_invalid_hook_block_is_rejected_by_validate_but_still_parses() {
        let config = memory_config(&spec_with(json!({
            "hooks": {"member_profile_importance": 1.5},
        })));
        let error = config.validate().unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{error}");
        assert!(
            error.to_string().contains("member_profile_importance"),
            "{error}"
        );
    }

    #[test]
    fn a_blank_store_dir_is_rejected_by_validate() {
        let config = MemoryConfig {
            store: MemoryStoreConfig::File {
                dir: Some(PathBuf::from("   ")),
            },
            ..MemoryConfig::default()
        };
        let error = config.validate().unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{error}");
        assert!(error.to_string().contains("dir"), "{error}");
        let blank = MemoryConfig {
            store: MemoryStoreConfig::File {
                dir: Some(PathBuf::from("")),
            },
            ..MemoryConfig::default()
        };
        assert!(blank.validate().is_err());
    }

    #[test]
    fn the_store_round_trips_through_json() {
        for store in [
            MemoryStoreConfig::Memory,
            MemoryStoreConfig::File { dir: None },
            MemoryStoreConfig::File {
                dir: Some(PathBuf::from("/var/lib/team")),
            },
        ] {
            let encoded = serde_json::to_value(&store).unwrap();
            let decoded: MemoryStoreConfig = serde_json::from_value(encoded.clone()).unwrap();
            assert_eq!(decoded, store, "for {encoded}");
        }
        let encoded = serde_json::to_value(MemoryStoreConfig::Memory).unwrap();
        assert_eq!(encoded, json!("memory"));
        let encoded = serde_json::to_value(MemoryStoreConfig::File { dir: None }).unwrap();
        assert_eq!(encoded, json!({"kind": "file"}));
    }

    #[test]
    fn the_whole_config_round_trips_through_json() {
        let mut spec = spec_with(json!({
            "enabled": false,
            "store": {"kind": "file", "dir": "/tmp/team"},
            "extractor": "llm",
            "export_on_stop": false,
            "import_on_build": false,
            "hooks": {"handoff_on_leave": false},
        }));
        spec.metadata.insert("other".to_owned(), json!(1));
        let config = memory_config(&spec);
        let encoded = serde_json::to_value(&config).unwrap();
        let decoded: MemoryConfig = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, config);
        // Unknown fields are tolerated on the way back in.
        let decoded: MemoryConfig = serde_json::from_value(json!({
            "enabled": false, "unknown_key": true,
        }))
        .unwrap();
        assert!(!decoded.enabled);
        assert_eq!(decoded.store, MemoryStoreConfig::Memory);
    }

    #[test]
    fn the_extractor_name_is_readable() {
        assert_eq!(MemoryExtractorKind::Rule.to_string(), "rule");
        assert_eq!(MemoryExtractorKind::Llm.to_string(), "llm");
        assert_eq!(MemoryExtractorKind::default(), MemoryExtractorKind::Rule);
    }
}
