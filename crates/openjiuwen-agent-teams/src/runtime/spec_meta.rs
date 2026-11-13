//! Spec-metadata configuration the runtime reads at build time.
//!
//! Everything the host tunes without touching the typed [`TeamAgentSpec`](crate::schema::TeamAgentSpec) surface lives under
//! [`TeamAgentSpec::metadata`](crate::schema::TeamAgentSpec::metadata): spawn mode, workspace isolation, the organisation layer, KV-cache
//! affinity and attach hosting. Parsing is deliberately tolerant — a value that does not decode is
//! logged and ignored (the in-process default stays in force), because a runtime must not fail just
//! because an optional knob is malformed. The organisation block is the exception: an explicitly
//! configured chart that does not validate fails the build.
//!
//! Recognised keys (all optional):
//!
//! | key | value | effect |
//! | --- | --- | --- |
//! | `spawn` | [`MemberSpawnOptions`](crate::member::MemberSpawnOptions) (`{"kind":"process", …}`) or `{"options": …, "fork": …}` | which spawner materialises a member |
//! | `fork` | [`ForkDirective`] | parent member + task override for a forked member |
//! | `workspace` | `{"isolation":"worktree"}` or `"worktree"` | workspace backend |
//! | `organization` | `{"spec": OrganizationSpec, "rules": DelegationRules}` | organisation layer |
//! | `kv_cache` | `{"enabled": true, "max_entries": 256}` | KV-cache affinity |
//! | `checkpoint` | [`MemberCheckpointPolicy`](crate::harness::MemberCheckpointPolicy) or `{"policy": …, "store": "memory"\|"file", "dir": "/abs/path"}` | per-member checkpoint capture/persistence |
//! | `attach` | `{"socket": "/run/team.sock", "required": true}` | Unix-socket attach host |
//!
//! `OPENJIUWEN_TEAM_SOCKET` is the environment fallback for `attach.socket`; an explicit metadata
//! socket always wins.
use std::path::PathBuf;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::warn;

use openjiuwen_agent::kv_cache::KVCacheAffinityConfig as KvCacheConfig;

use crate::harness::MemberCheckpointPolicy;
use crate::member::MemberSpawnOptions;
use crate::org::{DelegationRules, ForkTaskOverride, ForkToolPolicy, OrganizationSpec};
use crate::schema::{SpawnMode, TeamAgentSpec};
use crate::workspace::IsolationMode;

/// Metadata key holding the member spawn directive.
pub const SPAWN_METADATA_KEY: &str = "spawn";
/// Metadata key holding a fork directive (parent member + task override).
pub const FORK_METADATA_KEY: &str = "fork";
/// Metadata key holding workspace settings such as `isolation`.
pub const WORKSPACE_METADATA_KEY: &str = "workspace";
/// Metadata key holding the organisation block.
pub const ORGANIZATION_METADATA_KEY: &str = "organization";
/// Metadata key holding KV-cache affinity settings.
pub const KV_CACHE_METADATA_KEY: &str = "kv_cache";
/// Metadata key holding the per-member checkpoint configuration.
pub const CHECKPOINT_METADATA_KEY: &str = "checkpoint";
/// Metadata key holding the attach-host settings.
pub const ATTACH_METADATA_KEY: &str = "attach";
/// Environment variable naming the attach socket when the metadata does not.
pub const ATTACH_SOCKET_ENV: &str = "OPENJIUWEN_TEAM_SOCKET";

/// A fork directive: the member this spawn forks from and what the child starts on.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ForkDirective {
    /// Roster name of the parent member whose context is forked.
    pub parent: String,
    /// Task the child starts on instead of the parent's current task.
    pub task: Option<ForkTaskOverride>,
    /// Tool-set policy applied to the child.
    pub tools: ForkToolPolicy,
}
impl ForkDirective {
    /// Creates a directive that forks `parent` without a task or tool override.
    #[must_use]
    pub fn new(parent: impl Into<String>) -> Self {
        Self {
            parent: parent.into(),
            ..Self::default()
        }
    }
    /// Returns `true` when a parent member is named.
    #[must_use]
    pub fn is_configured(&self) -> bool {
        !self.parent.trim().is_empty()
    }
}

/// The parsed organisation block: the chart plus its delegation rules.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OrganizationBlock {
    /// Organisation chart (teams, members, reporting lines).
    pub spec: OrganizationSpec,
    /// Delegation rules layered over the chart.
    pub rules: DelegationRules,
}
impl OrganizationBlock {
    /// Returns `true` when the block names an organisation.
    #[must_use]
    pub fn is_configured(&self) -> bool {
        !self.spec.organization_id.trim().is_empty()
    }
}

/// The attach-host settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttachConfig {
    /// Unix domain socket the server binds.
    pub socket: PathBuf,
    /// Whether a bind failure must abort the run instead of being logged.
    pub required: bool,
}
impl AttachConfig {
    /// Creates a best-effort attach configuration for `socket`.
    #[must_use]
    pub fn new(socket: impl Into<PathBuf>) -> Self {
        Self {
            socket: socket.into(),
            required: false,
        }
    }
    /// Marks the socket as required: a bind failure fails the run.
    #[must_use]
    pub fn required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }
}

/// Reads the spawn directive from the spec metadata (default: in-process).
///
/// Two shapes are accepted: a bare [`MemberSpawnOptions`] object, or a wrapper
/// `{"options": …, "fork": …}` so a fork can travel with the spawn mode. `spawn_mode: process`
/// in the typed spec is honoured when the metadata names no spawn kind.
#[must_use]
pub fn spawn_options(spec: &TeamAgentSpec) -> MemberSpawnOptions {
    if let Some(value) = spec.metadata.get(SPAWN_METADATA_KEY) {
        let options = value.get("options").unwrap_or(value);
        match serde_json::from_value::<MemberSpawnOptions>(options.clone()) {
            Ok(options) => return options,
            Err(error) => warn!(
                team = %spec.team_name,
                %error,
                "ignoring malformed spawn metadata; falling back to in-process"
            ),
        }
    }
    if spec.spawn_mode == SpawnMode::Process {
        return MemberSpawnOptions::from_kind(crate::member::MemberSpawnKind::Process);
    }
    MemberSpawnOptions::default()
}

/// Reads the fork directive, if the spec configures one.
#[must_use]
pub fn fork_directive(spec: &TeamAgentSpec) -> Option<ForkDirective> {
    let from_spawn = spec
        .metadata
        .get(SPAWN_METADATA_KEY)
        .and_then(|value| value.get("fork"))
        .cloned();
    let value = from_spawn.or_else(|| spec.metadata.get(FORK_METADATA_KEY).cloned())?;
    match serde_json::from_value::<ForkDirective>(value) {
        Ok(directive) if directive.is_configured() => Some(directive),
        Ok(_) => None,
        Err(error) => {
            warn!(team = %spec.team_name, %error, "ignoring malformed fork metadata");
            None
        }
    }
}

/// Reads the requested workspace isolation (default [`IsolationMode::None`]).
#[must_use]
pub fn isolation_mode(spec: &TeamAgentSpec) -> IsolationMode {
    let Some(value) = spec.metadata.get(WORKSPACE_METADATA_KEY) else {
        return IsolationMode::None;
    };
    let raw = match value {
        Value::String(text) => Some(text.as_str()),
        Value::Object(map) => map.get("isolation").and_then(Value::as_str),
        _ => None,
    };
    match raw.map(str::trim) {
        Some(value) if value.eq_ignore_ascii_case("worktree") => IsolationMode::Worktree,
        Some(value) if value.is_empty() || value.eq_ignore_ascii_case("none") => {
            IsolationMode::None
        }
        Some(other) => {
            warn!(
                team = %spec.team_name,
                isolation = other,
                "unknown workspace isolation; using local directories"
            );
            IsolationMode::None
        }
        None => IsolationMode::None,
    }
}

/// Reads and validates the organisation block; `None` when no organisation is configured.
///
/// # Errors
///
/// Returns [`Error::Validation`] when the block exists but does not decode or the chart is
/// inconsistent (unknown reporting line, blank id, duplicate member). A configured-but-broken
/// organisation is a user error, so it fails the build rather than silently running flat.
pub fn organization_block(spec: &TeamAgentSpec) -> Result<Option<OrganizationBlock>> {
    let Some(value) = spec.metadata.get(ORGANIZATION_METADATA_KEY) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let block = serde_json::from_value::<OrganizationBlock>(value.clone()).map_err(|error| {
        Error::validation(format!(
            "team '{}' has an invalid organization block: {error}",
            spec.team_name
        ))
    })?;
    if !block.is_configured() {
        return Ok(None);
    }
    block.spec.validate()?;
    Ok(Some(block))
}

/// Reads the KV-cache affinity config (`enabled: true` by default, `max_entries` optional).
#[must_use]
pub fn kv_cache_config(spec: &TeamAgentSpec) -> KvCacheConfig {
    let mut config = KvCacheConfig {
        enabled: true,
        ..KvCacheConfig::default()
    };
    let Some(value) = spec.metadata.get(KV_CACHE_METADATA_KEY) else {
        return config;
    };
    match serde_json::from_value::<KvCacheConfig>(value.clone()) {
        Ok(parsed) => config = parsed,
        Err(error) => {
            warn!(team = %spec.team_name, %error, "ignoring malformed kv_cache metadata");
        }
    }
    config
}

/// Where a member's checkpoints are persisted.
///
/// `Memory` keeps them in the process (deterministic, lost on shutdown) and `File` writes one JSON
/// document per `(session id, card id)` under a directory. The directory is resolved per member, so
/// the same configuration gives every member its own subtree.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum CheckpointStoreConfig {
    /// Process-local store; the default, and the fallback for an unknown backend name.
    #[default]
    Memory,
    /// File backend; `dir = None` means `<member workspace root>/.openjiuwen/checkpoints`.
    File {
        /// Explicit root directory, when the metadata pins one.
        dir: Option<PathBuf>,
    },
}

/// The parsed `checkpoint` block: capture knobs plus the destination backend.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CheckpointConfig {
    /// Which boundaries capture a checkpoint and how a stored one is consumed.
    pub policy: MemberCheckpointPolicy,
    /// Where the records are persisted.
    pub store: CheckpointStoreConfig,
}

/// Reads the checkpoint configuration (default: the default policy on the in-memory backend).
///
/// Two shapes are accepted, both optional and both tolerant — a value that does not decode is
/// logged and leaves the default configuration in force:
///
/// * a bare [`MemberCheckpointPolicy`] (`{"enabled": false, "resume": "require"}`), which keeps the
///   in-memory backend; and
/// * a wrapper `{"policy": …, "store": "memory"|"file", "dir": "/abs/path"}` where `store` may also
///   be `{"kind": "file", "dir": "/abs/path"}`. A blank `dir` counts as absent, `store: "file"`
///   without a usable `dir` roots the store under the member's workspace, and a `dir` supplied
///   while the backend is `memory` is ignored with a warning. A `dir` with no `store` key selects
///   the file backend.
///
/// An absent key, `null` or a blank string is the default configuration; any other JSON type is
/// logged and ignored.
#[must_use]
pub fn checkpoint_config(spec: &TeamAgentSpec) -> CheckpointConfig {
    let Some(value) = spec.metadata.get(CHECKPOINT_METADATA_KEY) else {
        return CheckpointConfig::default();
    };
    match value {
        Value::Null => CheckpointConfig::default(),
        Value::String(text) if text.trim().is_empty() => CheckpointConfig::default(),
        Value::Object(map) => parse_checkpoint_block(spec, value, map),
        _ => {
            warn!(
                team = %spec.team_name,
                "ignoring checkpoint metadata that is not an object or a string; checkpointing \
                 stays at its default configuration"
            );
            CheckpointConfig::default()
        }
    }
}

/// Backend a `store` value names, before the directory is folded in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StoreKind {
    /// No `store` key (or a blank one): the backend follows the `dir` key.
    Unset,
    /// The in-memory backend.
    Memory,
    /// The file backend.
    File,
}

/// Parses one `checkpoint` object: the bare-policy shape or the `policy`/`store`/`dir` wrapper.
fn parse_checkpoint_block(
    spec: &TeamAgentSpec,
    value: &Value,
    map: &serde_json::Map<String, Value>,
) -> CheckpointConfig {
    if !["policy", "store", "dir"]
        .iter()
        .any(|key| map.contains_key(*key))
    {
        return match serde_json::from_value::<MemberCheckpointPolicy>(value.clone()) {
            Ok(policy) => CheckpointConfig {
                policy,
                store: CheckpointStoreConfig::Memory,
            },
            Err(error) => {
                warn!(team = %spec.team_name, %error, "ignoring malformed checkpoint metadata");
                CheckpointConfig::default()
            }
        };
    }
    let policy = match map.get("policy") {
        None | Some(Value::Null) => MemberCheckpointPolicy::default(),
        Some(policy) => match serde_json::from_value::<MemberCheckpointPolicy>(policy.clone()) {
            Ok(policy) => policy,
            Err(error) => {
                warn!(
                    team = %spec.team_name,
                    %error,
                    "ignoring malformed checkpoint policy; keeping the default policy"
                );
                MemberCheckpointPolicy::default()
            }
        },
    };
    let dir = parse_checkpoint_dir(spec, map.get("dir"));
    let (kind, store_dir) = match map.get("store") {
        None | Some(Value::Null) => (StoreKind::Unset, None),
        Some(Value::String(name)) => (store_kind(spec, name), None),
        Some(Value::Object(store)) => {
            let kind = store
                .get("kind")
                .map_or(StoreKind::Unset, |kind| match kind {
                    Value::String(name) => store_kind(spec, name),
                    _ => StoreKind::Unset,
                });
            (kind, parse_checkpoint_dir(spec, store.get("dir")))
        }
        Some(_) => {
            warn!(
                team = %spec.team_name,
                "ignoring malformed checkpoint store; keeping the in-memory backend"
            );
            (StoreKind::Unset, None)
        }
    };
    let dir = dir.or(store_dir);
    let store = match kind {
        StoreKind::File => CheckpointStoreConfig::File { dir },
        StoreKind::Memory => {
            if dir.is_some() {
                warn!(
                    team = %spec.team_name,
                    "ignoring the checkpoint dir: the in-memory backend persists nothing on disk"
                );
            }
            CheckpointStoreConfig::Memory
        }
        StoreKind::Unset => dir.map_or(CheckpointStoreConfig::Memory, |dir| {
            CheckpointStoreConfig::File { dir: Some(dir) }
        }),
    };
    CheckpointConfig { policy, store }
}

/// Resolves a `store` backend name; an unknown name warns and stays in memory.
fn store_kind(spec: &TeamAgentSpec, name: &str) -> StoreKind {
    let name = name.trim();
    if name.is_empty() {
        return StoreKind::Unset;
    }
    if name.eq_ignore_ascii_case("memory") {
        return StoreKind::Memory;
    }
    if name.eq_ignore_ascii_case("file") {
        return StoreKind::File;
    }
    warn!(
        team = %spec.team_name,
        store = name,
        "unknown checkpoint store; keeping the in-memory backend"
    );
    StoreKind::Memory
}

/// Reads an optional checkpoint directory: a blank string, `null` or a non-string is absent.
fn parse_checkpoint_dir(spec: &TeamAgentSpec, value: Option<&Value>) -> Option<PathBuf> {
    match value {
        None | Some(Value::Null) => None,
        Some(Value::String(path)) if path.trim().is_empty() => None,
        Some(Value::String(path)) => Some(PathBuf::from(path.trim())),
        Some(_) => {
            warn!(
                team = %spec.team_name,
                "ignoring a checkpoint dir that is not a path string"
            );
            None
        }
    }
}

/// Reads the attach-host configuration: metadata `attach` first, then [`ATTACH_SOCKET_ENV`].
/// An empty socket string is treated as "not configured". The metadata `required` flag defaults to
/// `false`, so a missing runtime directory only logs a warning.
#[must_use]
pub fn attach_config(spec: &TeamAgentSpec) -> Option<AttachConfig> {
    let mut socket: Option<String> = None;
    let mut required = false;
    if let Some(value) = spec.metadata.get(ATTACH_METADATA_KEY) {
        match value {
            Value::String(path) => socket = Some(path.clone()),
            Value::Object(map) => {
                socket = map
                    .get("socket")
                    .or_else(|| map.get("path"))
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                required = map
                    .get("required")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
            }
            _ => {}
        }
    }
    if socket.as_deref().is_none_or(|path| path.trim().is_empty()) {
        socket = std::env::var(ATTACH_SOCKET_ENV)
            .ok()
            .filter(|path| !path.trim().is_empty());
    }
    let socket = socket?;
    Some(AttachConfig {
        socket: PathBuf::from(socket.trim()),
        required,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Builds a team blueprint carrying one `checkpoint` metadata value.
    fn spec_with(value: Value) -> TeamAgentSpec {
        let mut spec = TeamAgentSpec {
            team_name: "alpha".to_owned(),
            ..TeamAgentSpec::default()
        };
        spec.metadata
            .insert(CHECKPOINT_METADATA_KEY.to_owned(), value);
        spec
    }

    #[test]
    fn absent_null_and_blank_values_keep_the_default_config() {
        let default = CheckpointConfig {
            policy: MemberCheckpointPolicy::default(),
            store: CheckpointStoreConfig::Memory,
        };
        let absent = TeamAgentSpec {
            team_name: "alpha".to_owned(),
            ..TeamAgentSpec::default()
        };
        assert_eq!(checkpoint_config(&absent), default);
        for value in [Value::Null, json!(""), json!("   ")] {
            assert_eq!(
                checkpoint_config(&spec_with(value.clone())),
                default,
                "{value}"
            );
        }
        assert!(default.policy.is_enabled());
    }

    #[test]
    fn a_bare_policy_object_selects_the_in_memory_backend() {
        let disabled = checkpoint_config(&spec_with(json!({"enabled": false})));
        assert!(!disabled.policy.is_enabled());
        assert!(disabled.policy.on_turn, "absent knobs keep their default");
        assert_eq!(disabled.store, CheckpointStoreConfig::Memory);

        let required = checkpoint_config(&spec_with(json!({"resume": "require"})));
        assert_eq!(
            required.policy.resume,
            crate::harness::CheckpointResume::Require
        );
        assert!(required.policy.is_enabled());
        assert_eq!(required.store, CheckpointStoreConfig::Memory);
    }

    #[test]
    fn a_wrapper_selects_the_file_backend_and_its_directory() {
        let config = checkpoint_config(&spec_with(json!({
            "policy": {"enabled": true, "on_state": true},
            "store": "file",
            "dir": "/var/lib/team/checkpoints",
        })));
        assert!(config.policy.on_state);
        assert_eq!(
            config.store,
            CheckpointStoreConfig::File {
                dir: Some(PathBuf::from("/var/lib/team/checkpoints")),
            }
        );
    }

    #[test]
    fn a_file_store_without_a_dir_defers_to_the_member_workspace() {
        for value in [
            json!({"store": "file"}),
            json!({"store": "file", "dir": "  "}),
            json!({"store": {"kind": "file"}}),
        ] {
            let config = checkpoint_config(&spec_with(value.clone()));
            assert_eq!(
                config.store,
                CheckpointStoreConfig::File { dir: None },
                "{value}"
            );
            assert!(config.policy.is_enabled());
        }
    }

    #[test]
    fn the_object_store_shape_carries_its_own_dir() {
        let config = checkpoint_config(&spec_with(json!({
            "store": {"kind": "file", "dir": "/tmp/team-cp"},
        })));
        assert_eq!(
            config.store,
            CheckpointStoreConfig::File {
                dir: Some(PathBuf::from("/tmp/team-cp")),
            }
        );
    }

    #[test]
    fn a_dir_without_a_store_key_selects_the_file_backend() {
        let config = checkpoint_config(&spec_with(json!({"dir": "/tmp/team-cp"})));
        assert_eq!(
            config.store,
            CheckpointStoreConfig::File {
                dir: Some(PathBuf::from("/tmp/team-cp")),
            }
        );
        assert_eq!(config.policy, MemberCheckpointPolicy::default());
    }

    #[test]
    fn a_policy_only_wrapper_keeps_the_memory_backend() {
        let config = checkpoint_config(&spec_with(json!({"policy": {"on_stop": false}})));
        assert!(!config.policy.on_stop);
        assert!(config.policy.on_turn);
        assert_eq!(config.store, CheckpointStoreConfig::Memory);
    }

    #[test]
    fn a_dir_on_the_memory_backend_is_ignored() {
        for value in [
            json!({"store": "memory", "dir": "/tmp/team-cp"}),
            json!({"store": {"kind": "memory", "dir": "/tmp/team-cp"}}),
        ] {
            let config = checkpoint_config(&spec_with(value.clone()));
            assert_eq!(config.store, CheckpointStoreConfig::Memory, "{value}");
        }
        let config = checkpoint_config(&spec_with(json!({"store": "memory"})));
        assert_eq!(config.store, CheckpointStoreConfig::Memory);
    }

    #[test]
    fn an_unknown_store_kind_stays_in_memory() {
        for value in [
            json!({"store": "redis"}),
            json!({"store": {"kind": "postgresql"}}),
        ] {
            let config = checkpoint_config(&spec_with(value.clone()));
            assert_eq!(config.store, CheckpointStoreConfig::Memory, "{value}");
            assert!(config.policy.is_enabled());
        }
    }

    #[test]
    fn a_malformed_wrapper_policy_falls_back_to_the_default_policy() {
        let config = checkpoint_config(&spec_with(json!({
            "policy": {"enabled": "not-a-bool"},
            "store": "file",
            "dir": "/tmp/team-cp",
        })));
        assert_eq!(config.policy, MemberCheckpointPolicy::default());
        assert_eq!(
            config.store,
            CheckpointStoreConfig::File {
                dir: Some(PathBuf::from("/tmp/team-cp")),
            }
        );
    }

    #[test]
    fn a_malformed_bare_policy_falls_back_to_the_default_config() {
        let config = checkpoint_config(&spec_with(json!({"enabled": {"nested": true}})));
        assert_eq!(config, CheckpointConfig::default());
    }

    #[test]
    fn non_object_values_fall_back_to_the_default_config() {
        for value in [json!(7), json!(["file"]), json!(true), json!("file")] {
            assert_eq!(
                checkpoint_config(&spec_with(value.clone())),
                CheckpointConfig::default(),
                "{value}"
            );
        }
    }

    #[test]
    fn a_non_string_dir_is_ignored() {
        let config = checkpoint_config(&spec_with(json!({"store": "file", "dir": 12})));
        assert_eq!(config.store, CheckpointStoreConfig::File { dir: None });
    }
}
