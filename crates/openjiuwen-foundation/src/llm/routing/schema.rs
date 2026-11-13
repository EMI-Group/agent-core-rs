//! Serializable payloads of the model-selection compiler and the routing model.
//!
//! Ports `openjiuwen/core/foundation/llm/routing/schema.py`. The layering is the
//! Python one: a persisted [`ModelSelection`] reference is resolved by an upper
//! layer into a [`ResolvedModel`] or a [`ResolvedModelGroup`], and
//! [`super::compiler::compile_model_selection`] turns that resolved payload into the
//! [`CompiledModelSelection`] pair of runtime configs.
//!
//! Two shapes deliberately differ from the Python DTOs, both documented on the
//! types themselves:
//!
//! - The routing policy of a group is **flattened** onto [`ResolvedModelGroup`]
//!   (`strategy`, `strategy_kwargs`, `num_retries`, `timeout_seconds`) instead of
//!   living in a nested `routing` object. The nested Python spelling is still
//!   accepted when deserializing, and the flat fields win when both are present.
//! - [`ResolvedRoute`] carries two additive fields — `priority` and `tag` — that
//!   implement the Rust rule/priority router (see
//!   [`super::compiler::resolve_model_group_routes`]). Python has no priorities;
//!   leaving them unset keeps the Python behaviour (declared list order).
//!
//! Everything here is data: no client, no router, and no I/O.
use std::fmt;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::security::SecretString;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use crate::llm::config::{ModelClientConfig, ModelRequestConfig};

/// Which kind of model selection a reference or compiled result describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionKind {
    /// One concrete model.
    Model,
    /// A group of candidate routes.
    ModelGroup,
}

impl SelectionKind {
    /// The `snake_case` wire name (`"model"` / `"model_group"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::ModelGroup => "model_group",
        }
    }
}

impl fmt::Display for SelectionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A persisted or user-provided reference to a model selection.
///
/// Mirrors the Python `ModelSelection`. The compiler never accepts this type: a
/// reference has to be resolved into a [`ResolvedModel`] or a
/// [`ResolvedModelGroup`] first (resolving needs ownership and persistence lookups
/// that live in upper layers). [`ModelSelectionWire::from_value`] rejects it with
/// a message saying exactly that.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ModelSelection {
    /// Whether [`id`](Self::id) names a model or a model group.
    #[serde(rename = "type")]
    pub kind: SelectionKind,
    /// Identifier of the referenced model or model group.
    pub id: String,
}

/// One concrete, callable model resolved by an upper layer.
///
/// Mirrors the Python `ResolvedModel`: identity, transport settings
/// (`provider`/`api_key`/`api_base`/`client_options`), and the request defaults a
/// route of this model starts from. The compiler maps this onto a
/// [`ModelClientConfig`] / [`ModelRequestConfig`] pair.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ResolvedModel {
    /// Identifier of the model record this route was resolved from.
    pub model_id: String,
    /// Provider-facing model name, for example `"gpt-4o"`.
    pub model_name: String,
    /// Provider name, for example `"openai"`; must not be blank.
    pub provider: String,
    /// API key; a blank string is treated as unset.
    #[serde(
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_opt_secret"
    )]
    pub api_key: Option<SecretString>,
    /// Provider endpoint override; a blank string is treated as unset.
    #[serde(
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_opt_text"
    )]
    pub api_base: Option<String>,
    /// Where the resolution came from (Python default `"defaults"`); informational.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Endpoint profile name, for example `"deepseek"`; travels in
    /// [`ModelClientConfig::extra`] because the Rust client config does not model it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint_profile: Option<String>,
    /// Tag a `tag-filtered` strategy can pin a route by.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_tag: Option<String>,
    /// Human-readable description, used in logs and diagnostics only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_description: Option<String>,
    /// Client-side knobs; recognized keys map onto [`ModelClientConfig`] fields and
    /// every other key is preserved in [`ModelClientConfig::extra`].
    pub client_options: Value,
    /// Request-level defaults; merged below the group and route layers.
    pub request_defaults: Value,
}

/// One candidate route inside a [`ResolvedModelGroup`].
///
/// Mirrors the Python `ResolvedRoute` plus the two additive Rust fields `priority`
/// and `tag`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ResolvedRoute {
    /// Identifier of this route inside the group.
    pub route_id: String,
    /// The model this route calls.
    pub model: ResolvedModel,
    /// Whether the route participates in routing; disabled routes are ignored.
    pub enabled: bool,
    /// Request values merged **above** the group config.
    pub request_overrides: Value,
    /// Tokens-per-minute hint from the resolved record; not enforced (no limiter).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tpm: Option<f64>,
    /// Requests-per-minute hint from the resolved record; not enforced (no limiter).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rpm: Option<f64>,
    /// Per-route timeout in seconds; applied to the route client's request timeout.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<f64>,
    /// Additive Rust field: higher priorities are tried first.
    ///
    /// The priority layer only engages when at least one route declares a priority;
    /// see [`super::compiler::resolve_model_group_routes`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<i64>,
    /// Additive Rust field: an alternative tag a `tag-filtered` strategy matches.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
}

impl Default for ResolvedRoute {
    /// An enabled route with an empty id, a default model, and no hints.
    fn default() -> Self {
        Self {
            route_id: String::new(),
            model: ResolvedModel::default(),
            enabled: true,
            request_overrides: Value::Object(Map::new()),
            tpm: None,
            rpm: None,
            timeout: None,
            priority: None,
            tag: None,
        }
    }
}

/// A resolved model group ready for compilation and routing.
///
/// Mirrors the Python `ResolvedModelGroup`, with the routing policy flattened onto
/// the group (see the module docs). `strategy` being `None` — or an empty string —
/// means "no routing": the group compiles to the config of its first enabled
/// route and behaves exactly like a single model.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ResolvedModelGroup {
    /// Identifier of the group; the routing model reports it as its model name.
    pub group_id: String,
    /// Candidate routes in declared order (the routing order, before priorities).
    pub routes: Vec<ResolvedRoute>,
    /// Group-level request defaults, merged above every route's own defaults.
    #[serde(skip_serializing_if = "Value::is_null")]
    pub request_config: Value,
    /// Routing strategy name; `None` means no routing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strategy: Option<String>,
    /// Strategy arguments, for example `{"tag": "east"}` for `tag-filtered`.
    #[serde(skip_serializing_if = "Value::is_null")]
    pub strategy_kwargs: Value,
    /// Additional attempts after the first one; `None` defaults to
    /// `enabled_routes - 1`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_retries: Option<u32>,
    /// Whole-group deadline in seconds; recorded in the compiled plan, not enforced.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_seconds: Option<f64>,
    /// Default tag for a `tag-filtered` strategy when `strategy_kwargs` names none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_tag: Option<String>,
}

impl Default for ResolvedModelGroup {
    /// An empty group with no strategy and empty JSON maps.
    fn default() -> Self {
        Self {
            group_id: String::new(),
            routes: Vec::new(),
            request_config: Value::Object(Map::new()),
            strategy: None,
            strategy_kwargs: Value::Object(Map::new()),
            num_retries: None,
            timeout_seconds: None,
            fallback_tag: None,
        }
    }
}

impl ResolvedModelGroup {
    /// The routing policy of this group with defaults resolved.
    ///
    /// [`ResolvedRoutingConfig::strategy`] always carries a strategy, so this
    /// accessor reports the policy that *would* apply; use
    /// [`strategy`](Self::strategy) (or
    /// [`super::compiler::compile_model_selection`]) to learn whether the group
    /// routes at all. `num_retries` defaults to `enabled_routes - 1`, matching the
    /// compiler's rule.
    #[must_use]
    pub fn routing(&self) -> ResolvedRoutingConfig {
        let strategy = self
            .strategy
            .as_deref()
            .map(RoutingStrategy::parse)
            .unwrap_or_default();
        let enabled = self.routes.iter().filter(|route| route.enabled).count();
        let num_retries = self
            .num_retries
            .unwrap_or_else(|| u32::try_from(enabled.saturating_sub(1)).unwrap_or(u32::MAX));
        ResolvedRoutingConfig {
            strategy,
            num_retries,
            strategy_kwargs: self.strategy_kwargs.clone(),
            timeout_seconds: self.timeout_seconds,
        }
    }
}

/// Deserialization mirror of [`ResolvedModelGroup`].
///
/// Accepts both the flattened Rust shape and the nested Python shape
/// (`{"routing": {"strategy": ...}}`); flat fields win when both are present.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case", default)]
struct ResolvedModelGroupWire {
    #[serde(alias = "model_group_id")]
    group_id: String,
    routes: Vec<ResolvedRoute>,
    request_config: Value,
    strategy: Option<String>,
    strategy_kwargs: Value,
    num_retries: Option<u32>,
    timeout_seconds: Option<f64>,
    fallback_tag: Option<String>,
    routing: Option<NestedRoutingWire>,
}

/// The nested Python `routing` object of a model group.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case", default)]
struct NestedRoutingWire {
    strategy: Option<String>,
    strategy_kwargs: Value,
    num_retries: Option<u32>,
    timeout_seconds: Option<f64>,
}

impl From<ResolvedModelGroupWire> for ResolvedModelGroup {
    fn from(wire: ResolvedModelGroupWire) -> Self {
        let nested = wire.routing.unwrap_or_default();
        let strategy_kwargs = if wire.strategy_kwargs.is_null() {
            nested.strategy_kwargs
        } else {
            wire.strategy_kwargs
        };
        Self {
            group_id: wire.group_id,
            routes: wire.routes,
            request_config: wire.request_config,
            strategy: wire.strategy.or(nested.strategy),
            strategy_kwargs,
            num_retries: wire.num_retries.or(nested.num_retries),
            timeout_seconds: wire.timeout_seconds.or(nested.timeout_seconds),
            fallback_tag: wire.fallback_tag,
        }
    }
}

impl<'de> Deserialize<'de> for ResolvedModelGroup {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        ResolvedModelGroupWire::deserialize(deserializer).map(Self::from)
    }
}

/// The routing policy selected for a resolved model group.
///
/// Mirrors the Python `ResolvedRoutingConfig`, restricted to the fields the Rust
/// router implements: health-checking and observability toggles live only in the
/// (unported) `intelli_router` package.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ResolvedRoutingConfig {
    /// Strategy to apply; unknown names survive as [`RoutingStrategy::Custom`] and
    /// are rejected by the compiler.
    pub strategy: RoutingStrategy,
    /// Additional attempts after the first one.
    pub num_retries: u32,
    /// Strategy arguments; an empty object means "no arguments".
    pub strategy_kwargs: Value,
    /// Whole-group deadline in seconds; recorded, not enforced.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_seconds: Option<f64>,
}

impl Default for ResolvedRoutingConfig {
    /// `ordered-failover` with no retries, no arguments, and no deadline.
    fn default() -> Self {
        Self {
            strategy: RoutingStrategy::OrderedFailover,
            num_retries: 0,
            strategy_kwargs: Value::Object(Map::new()),
            timeout_seconds: None,
        }
    }
}

/// A model-group routing strategy.
///
/// Only the two Python strategies have dedicated behaviour; every other name is
/// preserved as [`RoutingStrategy::Custom`] so a payload round-trips, and
/// [`super::compiler::compile_model_selection`] rejects it as a validation error.
/// The routing model treats a `Custom` strategy as `ordered-failover`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RoutingStrategy {
    /// Try the candidates in order, falling over on retryable failures.
    OrderedFailover,
    /// Pin the candidates to one `fallback_tag` first.
    TagFiltered,
    /// A strategy name this implementation does not implement.
    Custom(String),
}

impl RoutingStrategy {
    /// Wire name of [`RoutingStrategy::OrderedFailover`].
    pub const ORDERED_FAILOVER: &'static str = "ordered-failover";
    /// Wire name of [`RoutingStrategy::TagFiltered`].
    pub const TAG_FILTERED: &'static str = "tag-filtered";

    /// Parses a strategy name leniently (trimmed, case- and separator-insensitive).
    ///
    /// This never fails: an unrecognized name becomes
    /// [`RoutingStrategy::Custom`], which the compiler then rejects with a
    /// validation error naming the supported strategies.
    #[must_use]
    pub fn parse(raw: &str) -> Self {
        let normalized = normalize_name(raw);
        match normalized.as_str() {
            Self::ORDERED_FAILOVER | "orderedfailover" => Self::OrderedFailover,
            Self::TAG_FILTERED | "tagfiltered" => Self::TagFiltered,
            _ => Self::Custom(normalized),
        }
    }

    /// The wire name of this strategy.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::OrderedFailover => Self::ORDERED_FAILOVER,
            Self::TagFiltered => Self::TAG_FILTERED,
            Self::Custom(name) => name,
        }
    }

    /// Whether this implementation gives the strategy dedicated behaviour.
    #[must_use]
    pub fn is_known(&self) -> bool {
        matches!(self, Self::OrderedFailover | Self::TagFiltered)
    }
}

impl Default for RoutingStrategy {
    /// [`RoutingStrategy::OrderedFailover`].
    fn default() -> Self {
        Self::OrderedFailover
    }
}

impl fmt::Display for RoutingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for RoutingStrategy {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for RoutingStrategy {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Ok(Self::parse(&raw))
    }
}

/// The two configs `Model` needs after compiling a selection.
///
/// Mirrors the Python `CompiledModelSelection`: `selected_type`/`selected_id` echo
/// which branch of [`super::compiler::compile_model_selection`] ran.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CompiledModelSelection {
    /// Transport configuration the caller builds a model from.
    pub model_client_config: ModelClientConfig,
    /// Request configuration the caller passes to `invoke`/`stream`, or uses as
    /// defaults when it calls with its own request.
    pub model_request_config: ModelRequestConfig,
    /// Whether the compiled payload was a model or a model group.
    pub selected_type: SelectionKind,
    /// The `model_id` or `group_id` the payload was compiled from.
    pub selected_id: String,
}

/// A JSON-shaped model selection handed to the compiler.
///
/// The Python compiler discriminates a mapping by the presence of the `"routes"`
/// key: a payload with routes is a model group, anything else is a model. This
/// enum does the same and additionally rejects a persisted
/// [`ModelSelection`] (`{"type": ..., "id": ...}`), which has to be resolved first.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ModelSelectionWire {
    /// A single resolved model.
    Model(ResolvedModel),
    /// A resolved model group.
    ModelGroup(ResolvedModelGroup),
}

impl ModelSelectionWire {
    /// Classifies an already-decoded JSON payload.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the value is not a JSON object, when it
    /// looks like an unresolved [`ModelSelection`], or when the payload does not
    /// match the shape its discrimination key selects.
    pub fn from_value(value: Value) -> Result<Self> {
        if !value.is_object() {
            return Err(Error::validation(format!(
                "model selection payload must be a JSON object, found {}",
                json_kind(&value)
            )));
        }
        if value.get("routes").is_some() {
            let group: ResolvedModelGroup = serde_json::from_value(value).map_err(|error| {
                Error::validation(format!("invalid model group payload: {error}"))
            })?;
            return Ok(Self::ModelGroup(group));
        }
        if value.get("model_id").is_none()
            && value.get("model_name").is_none()
            && value.get("type").is_some()
        {
            let kind = value
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("model")
                .to_owned();
            let id = value
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            return Err(Error::validation(format!(
                "model selection (type={kind}, id={id}) is not resolved; pass a resolved \
                 model or model group payload"
            )));
        }
        let model: ResolvedModel = serde_json::from_value(value)
            .map_err(|error| Error::validation(format!("invalid model payload: {error}")))?;
        Ok(Self::Model(model))
    }

    /// The model payload, when this selection is a single model.
    #[must_use]
    pub fn as_model(&self) -> Option<&ResolvedModel> {
        match self {
            Self::Model(model) => Some(model),
            Self::ModelGroup(_) => None,
        }
    }

    /// The model-group payload, when this selection is a group.
    #[must_use]
    pub fn as_group(&self) -> Option<&ResolvedModelGroup> {
        match self {
            Self::Model(_) => None,
            Self::ModelGroup(group) => Some(group),
        }
    }

    /// Which branch of the compiler this selection takes.
    #[must_use]
    pub fn kind(&self) -> SelectionKind {
        match self {
            Self::Model(_) => SelectionKind::Model,
            Self::ModelGroup(_) => SelectionKind::ModelGroup,
        }
    }
}

impl From<ResolvedModel> for ModelSelectionWire {
    fn from(model: ResolvedModel) -> Self {
        Self::Model(model)
    }
}

impl From<ResolvedModelGroup> for ModelSelectionWire {
    fn from(group: ResolvedModelGroup) -> Self {
        Self::ModelGroup(group)
    }
}

impl<'de> Deserialize<'de> for ModelSelectionWire {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        Self::from_value(value).map_err(<D::Error as serde::de::Error>::custom)
    }
}

/// Normalizes a strategy or provider name: trimmed, lowercased, `_`/` ` → `-`.
pub(crate) fn normalize_name(raw: &str) -> String {
    raw.trim().to_lowercase().replace(['_', ' '], "-")
}

/// The JSON kind of a value, for validation messages.
pub(crate) fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Deserializes an API key, mapping a blank string to "unset".
fn deserialize_opt_secret<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<SecretString>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<SecretString>::deserialize(deserializer)?;
    Ok(value.filter(|secret| !secret.expose_secret().trim().is_empty()))
}

/// Deserializes a text setting, mapping a blank string to "unset".
fn deserialize_opt_text<'de, D>(deserializer: D) -> std::result::Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<String>::deserialize(deserializer)?;
    Ok(value.filter(|text| !text.trim().is_empty()))
}
