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
//!   "trace": true,
//!   "hooks": {"member_profile_on_join": true, "handoff_importance": 0.6}
//! }
//! ```
//!
//! The `hooks` block is deserialized straight into
//! [`TeamMemoryHookConfig`] from `openjiuwen_memory::team`, so every switch and importance the
//! engine understands is reachable from metadata without a second schema here.
//!
//! `trace` is the only observability switch: it selects whether the wiring attaches the crate's
//! team-memory observer to the facade (see
//! [`trace`](crate::memory::trace)). The in-memory span sink
//! ([`MemoryConfig::otel_sink`]) and the caller-supplied observer
//! ([`MemoryConfig::observer`]) are process-local handles, so neither is part of the metadata
//! schema and neither is ever serialized.
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_memory::team::{TeamMemoryHookConfig, TeamMemoryObserver};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use tracing::warn;

use crate::memory::trace::OtelSpanSink;
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
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum MemoryStoreConfig {
    /// The process-local in-memory store.
    #[default]
    Memory,
    /// A durable store rooted at `dir`; `None` keeps the workspace root.
    File {
        /// Directory holding the `TEAM_MEMORY.md` document.
        dir: Option<PathBuf>,
    },
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
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryExtractorKind {
    /// The deterministic rule extractor (the default; no model call).
    #[default]
    Rule,
    /// The model-backed extractor, which degrades to the rules on failure.
    Llm,
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
/// build, every operation traced).
///
/// `Debug` and `PartialEq` are hand-written because the `otel` sink and the observer are shared
/// handles: each is reported as "present" rather than dumped, and two configurations are equal
/// only when they hold the *same* handle ([`Arc::ptr_eq`]).
///
/// The booleans are independent switches for unrelated parts of the wiring (the master switch, the
/// two document passes, tracing), not a state machine, so grouping them into enums would flatten
/// the configuration shape this type mirrors.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Serialize, Deserialize)]
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
    /// Trace every team-memory operation through the crate's observability seam.
    ///
    /// When `true` (the default) the wiring attaches a
    /// [`MemorySpanObserver`](crate::observability::MemorySpanObserver) to the facade, so each
    /// operation becomes one `memory.{operation}` span plus one structured `tracing` event. A
    /// disabled configuration attaches no observer whatever this says, and `false` attaches
    /// nothing at all — not even a caller-supplied [`MemoryConfig::observer`].
    pub trace: bool,
    /// The in-memory span sink the observer also writes to on the `otel` build.
    ///
    /// Never serialized: a sink is a process-local handle and metadata cannot name one. The field
    /// exists on every build, so the struct shape does not change with features; without `otel`
    /// the sink type is uninhabited and the slot always stays `None`.
    #[serde(skip)]
    pub otel_sink: Option<Arc<OtelSpanSink>>,
    /// A caller-supplied observer, which wins over the built-in
    /// [`MemorySpanObserver`](crate::observability::MemorySpanObserver) bridge.
    ///
    /// The field is the host's way to attach its own sink — in particular the OTLP bridge
    /// (`OtlpMemoryObserver`, feature `otlp`) — without rebuilding the facade itself. It is applied
    /// by [`observer_for`](crate::memory::trace::observer_for) where the facade is built, because
    /// [`TeamMemory::with_observer`](openjiuwen_memory::team::TeamMemory::with_observer) consumes
    /// the facade; [`MemoryConfig::trace`] still gates it, so a `trace: false` wiring attaches
    /// nothing even when an observer is supplied.
    ///
    /// Never serialized and never parsed from metadata: an observer is a process-local handle, and
    /// [`memory_config`] always sets it to `None`. The field exists on every build, like
    /// [`MemoryConfig::otel_sink`].
    #[serde(skip)]
    pub observer: Option<Arc<dyn TeamMemoryObserver>>,
    /// The lifecycle-hook switches and importances handed to the engine.
    pub hooks: TeamMemoryHookConfig,
}

impl fmt::Debug for MemoryConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MemoryConfig")
            .field("enabled", &self.enabled)
            .field("store", &self.store)
            .field("extractor", &self.extractor)
            .field("export_on_stop", &self.export_on_stop)
            .field("import_on_build", &self.import_on_build)
            .field("trace", &self.trace)
            .field("has_otel_sink", &self.otel_sink.is_some())
            .field("has_observer", &self.observer.is_some())
            .field("hooks", &self.hooks)
            .finish()
    }
}

impl PartialEq for MemoryConfig {
    fn eq(&self, other: &Self) -> bool {
        self.enabled == other.enabled
            && self.store == other.store
            && self.extractor == other.extractor
            && self.export_on_stop == other.export_on_stop
            && self.import_on_build == other.import_on_build
            && self.trace == other.trace
            && self.hooks == other.hooks
            && match (self.otel_sink.as_ref(), other.otel_sink.as_ref()) {
                (None, None) => true,
                (Some(left), Some(right)) => Arc::ptr_eq(left, right),
                _ => false,
            }
            && match (self.observer.as_ref(), other.observer.as_ref()) {
                (None, None) => true,
                (Some(left), Some(right)) => Arc::ptr_eq(left, right),
                _ => false,
            }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            store: MemoryStoreConfig::default(),
            extractor: MemoryExtractorKind::default(),
            export_on_stop: true,
            import_on_build: true,
            trace: true,
            otel_sink: None,
            observer: None,
            hooks: TeamMemoryHookConfig::default(),
        }
    }
}

impl MemoryConfig {
    /// Validates the hook switches and the store directory.
    ///
    /// [`MemoryConfig::otel_sink`] and [`MemoryConfig::observer`] are deliberately not validated:
    /// they are process-local handles rather than user input, every build carries both fields, and
    /// without the `otel` feature the sink can never be `Some`.
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
        trace: parse_bool(spec, map, "trace", default.trace),
        otel_sink: None,
        observer: None,
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
        assert!(config.trace);
        assert!(config.otel_sink.is_none());
        assert_eq!(config.hooks, TeamMemoryHookConfig::default());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn trace_is_read_from_the_metadata_and_defaults_to_on() {
        assert!(!memory_config(&spec_with(json!({"trace": false}))).trace);
        assert!(memory_config(&spec_with(json!({"trace": true}))).trace);
        assert!(memory_config(&spec_with(json!({"enabled": true}))).trace);
        // A malformed switch keeps the default: tracing stays on.
        for value in [json!("off"), json!(0), json!([])] {
            assert!(
                memory_config(&spec_with(json!({"trace": value}))).trace,
                "{value}"
            );
        }
    }

    #[test]
    fn a_bare_object_is_read_key_by_key() {
        let config = memory_config(&spec_with(json!({
            "enabled": false,
            "store": "memory",
            "extractor": "llm",
            "export_on_stop": false,
            "import_on_build": false,
            "trace": false,
            "hooks": {"member_profile_on_join": false, "handoff_importance": 0.9},
        })));
        assert!(!config.enabled);
        assert_eq!(config.store, MemoryStoreConfig::Memory);
        assert_eq!(config.extractor, MemoryExtractorKind::Llm);
        assert!(!config.export_on_stop);
        assert!(!config.import_on_build);
        assert!(!config.trace);
        assert!(config.otel_sink.is_none());
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
            "trace": false,
            "hooks": {"handoff_on_leave": false},
        }));
        spec.metadata.insert("other".to_owned(), json!(1));
        let config = memory_config(&spec);
        assert!(!config.trace);
        let encoded = serde_json::to_value(&config).unwrap();
        let decoded: MemoryConfig = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, config);
        assert!(!decoded.trace);
        // Unknown fields are tolerated on the way back in.
        let decoded: MemoryConfig = serde_json::from_value(json!({
            "enabled": false, "unknown_key": true,
        }))
        .unwrap();
        assert!(!decoded.enabled);
        assert_eq!(decoded.store, MemoryStoreConfig::Memory);
        assert!(decoded.trace);
    }

    #[test]
    fn the_sink_slot_is_never_serialized() {
        let encoded = serde_json::to_value(MemoryConfig::default()).unwrap();
        let object = encoded.as_object().unwrap();
        assert!(!object.contains_key("otel_sink"), "{encoded}");
        assert_eq!(object.get("trace"), Some(&json!(true)));
        // A configuration decoded from JSON carries no sink.
        let decoded: MemoryConfig = serde_json::from_value(encoded).unwrap();
        assert!(decoded.otel_sink.is_none());
    }

    #[test]
    fn the_observer_slot_is_never_serialized() {
        let encoded = serde_json::to_value(MemoryConfig::default()).unwrap();
        let object = encoded.as_object().unwrap();
        assert!(!object.contains_key("observer"), "{encoded}");
        // A configuration parsed from metadata carries no observer either.
        let spec = spec_with(json!({"trace": true}));
        assert!(memory_config(&spec).observer.is_none());
        let decoded: MemoryConfig = serde_json::from_value(encoded).unwrap();
        assert!(decoded.observer.is_none());
    }

    /// A no-op observer, so the slot can be populated without a sink.
    struct SilentObserver;

    impl openjiuwen_memory::team::TeamMemoryObserver for SilentObserver {
        fn observe(&self, _event: &openjiuwen_memory::team::TeamMemoryEvent) {}
    }

    #[test]
    fn the_hand_written_debug_and_partial_eq_cover_the_observer() {
        let first: Arc<dyn openjiuwen_memory::team::TeamMemoryObserver> = Arc::new(SilentObserver);
        let configured = MemoryConfig {
            observer: Some(Arc::clone(&first)),
            ..MemoryConfig::default()
        };
        let rendered = format!("{configured:?}");
        assert!(rendered.contains("has_observer: true"), "{rendered}");
        assert!(
            !rendered.contains("SilentObserver"),
            "the observer is reported as a flag, not dumped: {rendered}"
        );
        // Cloning keeps the same handle, so the two configurations are equal.
        assert_eq!(configured, configured.clone());
        // A different observer is a different configuration, even of the same type.
        let other = MemoryConfig {
            observer: Some(Arc::new(SilentObserver)),
            ..MemoryConfig::default()
        };
        assert_ne!(configured, other);
        // None versus Some is never equal, and the default has none.
        assert_ne!(configured, MemoryConfig::default());
        assert!(MemoryConfig::default().observer.is_none());
        assert!(format!("{:?}", MemoryConfig::default()).contains("has_observer: false"));
    }

    #[test]
    fn equality_ignores_nothing_but_compares_the_other_fields() {
        let base = MemoryConfig::default();
        assert_eq!(base, MemoryConfig::default());
        for other in [
            MemoryConfig {
                enabled: false,
                ..MemoryConfig::default()
            },
            MemoryConfig {
                trace: false,
                ..MemoryConfig::default()
            },
            MemoryConfig {
                export_on_stop: false,
                ..MemoryConfig::default()
            },
            MemoryConfig {
                import_on_build: false,
                ..MemoryConfig::default()
            },
            MemoryConfig {
                extractor: MemoryExtractorKind::Llm,
                ..MemoryConfig::default()
            },
            MemoryConfig {
                store: MemoryStoreConfig::File { dir: None },
                ..MemoryConfig::default()
            },
        ] {
            assert_ne!(base, other);
        }
        // The hand-written `Debug` reports the sink as a flag, never its contents.
        let rendered = format!("{base:?}");
        assert!(rendered.contains("has_otel_sink: false"), "{rendered}");
        assert!(rendered.contains("trace: true"), "{rendered}");
    }

    #[cfg(feature = "otel")]
    #[test]
    fn two_configurations_share_a_sink_only_by_pointer() {
        let sink = Arc::new(OtelSpanSink::new(4));
        let traced = MemoryConfig {
            otel_sink: Some(Arc::clone(&sink)),
            ..MemoryConfig::default()
        };
        let same_sink = traced.clone();
        assert_eq!(traced, same_sink);
        assert!(format!("{traced:?}").contains("has_otel_sink: true"));
        let other_sink = MemoryConfig {
            otel_sink: Some(Arc::new(OtelSpanSink::new(4))),
            ..MemoryConfig::default()
        };
        assert_ne!(traced, other_sink);
        assert_ne!(traced, MemoryConfig::default());
    }

    #[test]
    fn the_extractor_name_is_readable() {
        assert_eq!(MemoryExtractorKind::Rule.to_string(), "rule");
        assert_eq!(MemoryExtractorKind::Llm.to_string(), "llm");
        assert_eq!(MemoryExtractorKind::default(), MemoryExtractorKind::Rule);
    }
}
