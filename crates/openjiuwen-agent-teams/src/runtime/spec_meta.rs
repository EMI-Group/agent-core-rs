//! Spec-metadata configuration the runtime reads at build time.
//!
//! Everything the host tunes without touching the typed [`TeamAgentSpec`] surface lives under
//! [`TeamAgentSpec::metadata`]: spawn mode, workspace isolation, the organisation layer, KV-cache
//! affinity and attach hosting. Parsing is deliberately tolerant — a value that does not decode is
//! logged and ignored (the in-process default stays in force), because a runtime must not fail just
//! because an optional knob is malformed. The organisation block is the exception: an explicitly
//! configured chart that does not validate fails the build.
//!
//! Recognised keys (all optional):
//!
//! | key | value | effect |
//! | --- | --- | --- |
//! | `spawn` | [`MemberSpawnOptions`] (`{"kind":"process", …}`) or `{"options": …, "fork": …}` | which spawner materialises a member |
//! | `fork` | [`ForkDirective`] | parent member + task override for a forked member |
//! | `workspace` | `{"isolation":"worktree"}` or `"worktree"` | workspace backend |
//! | `organization` | `{"spec": OrganizationSpec, "rules": DelegationRules}` | organisation layer |
//! | `kv_cache` | `{"enabled": true, "max_entries": 256}` | KV-cache affinity |
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

/// Reads the attach-host configuration: metadata `attach` first, then [`ATTACH_SOCKET_ENV`].
///
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
