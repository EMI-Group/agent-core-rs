//! Model-selection compiler: resolved payload → runtime `Model` configuration.
//!
//! Ports `openjiuwen/core/foundation/llm/routing/compiler.py`. The compiler is pure:
//! it validates an already-resolved payload and converts it into the
//! [`ModelClientConfig`] / [`ModelRequestConfig`] pair a model consumes. It never
//! reads a database, a configuration file, or the network, and it resolves no ids —
//! ownership, permissions, and persistence lookups stay in the calling layer.
//!
//! Three entry points matter:
//!
//! - [`compile_model_selection`] — the Python `compile_model_selection`.
//! - [`compile_route`] — one enabled route of a group (used by the routing model
//!   and by callers that want per-route configs).
//! - [`resolve_model_group_routes`] — the enabled routes in routing order, with the
//!   additive priority rule layer applied.
//!
//! Request-default merging follows the Python precedence exactly:
//! `route.model.request_defaults` < group `request_config` <
//! `route.request_overrides`. Later sources win, nested objects merge recursively,
//! a `null` never unsets an earlier value, and keys that belong to the client
//! config or that the compiler owns are stripped (see [`merge_request_defaults`]).
//!
//! Two deliberate adaptations of the Python behaviour:
//!
//! - In router mode the Python compiler builds an `IntelliRouterConfig` inside
//!   `ModelClientConfig.intelli_router`. Here the compiled client config keeps the
//!   router identity (`provider = "intelli_router"`) and records the resolved plan
//!   as JSON in [`ModelClientConfig::extra`] under `"intelli_router"`; the
//!   executable path is [`super::model::RoutingModel`], which builds one real model
//!   client per route.
//! - A route's `reasoning` intent is not resolved here. It travels through
//!   [`ModelRequestConfig::extra`], where the OpenAI-compatible request builder
//!   resolves it through `openjiuwen_foundation::llm::reasoning` — the same
//!   provider-neutral intent, applied one layer lower.
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use serde_json::{Map, Value};

use super::schema::{
    json_kind, CompiledModelSelection, ModelSelectionWire, ResolvedModel, ResolvedModelGroup,
    ResolvedRoute, RoutingStrategy, SelectionKind,
};
use crate::llm::config::{ModelClientConfig, ModelRequestConfig};

/// Client provider name recorded when a model group compiles in router mode.
pub const INTELLI_ROUTER_PROVIDER: &str = "intelli_router";

/// Key of the routing plan inside a router-mode [`ModelClientConfig::extra`].
pub const ROUTING_PLAN_KEY: &str = "intelli_router";

/// Request keys the compiler owns or that belong to the client configuration.
///
/// They are stripped from every merged request-default layer: a request config can
/// never name the model, carry a transport credential, or flip streaming (the
/// caller decides how to call the model).
const NON_REQUEST_KEYS: &[&str] = &[
    "api_key",
    "api_base",
    "client_id",
    "client_provider",
    "deployment",
    "endpoint_profile",
    "headers",
    "max_retries",
    "messages",
    "model",
    "model_name",
    "provider",
    "stream",
    "stream_first_chunk_timeout",
    "stream_idle_timeout",
    "timeout",
    "verify_ssl",
];

/// Compiles a resolved model selection into the two runtime configs.
///
/// The branch is chosen by the payload shape (see [`ModelSelectionWire`]):
///
/// - **Model** — validates `model_name` and `provider`, then builds the client
///   config and a request config whose `model_name` is the model's name and whose
///   sampling settings come from the model's `request_defaults`.
/// - **Model group without a strategy** — behaves like a model: the group collapses
///   to the config of its **first enabled route**, with the group's
///   `request_config` and the route's `request_overrides` merged in.
/// - **Model group with a strategy** — validates the strategy name and its `tag`
///   argument, defaults `num_retries` to `enabled_routes - 1`, and returns a client
///   config that records the router identity plus the resolved plan in
///   [`ModelClientConfig::extra`]. The request config carries the group id as
///   `model_name` and the minimum positive `context_window` of the enabled routes
///   in `extra` (the Python compiler's only surviving group request setting).
///
/// # Errors
///
/// Returns [`Error::Validation`] when a model has a blank `model_name` or
/// `provider`, when a group has no enabled route, when the strategy name is
/// unknown, when a `tag` argument has the wrong type, or when a client option has
/// the wrong type.
pub fn compile_model_selection(selection: &ModelSelectionWire) -> Result<CompiledModelSelection> {
    match selection {
        ModelSelectionWire::Model(model) => {
            let (client, request) = compile_model(model)?;
            Ok(CompiledModelSelection {
                model_client_config: client,
                model_request_config: request,
                selected_type: SelectionKind::Model,
                selected_id: model.model_id.clone(),
            })
        }
        ModelSelectionWire::ModelGroup(group) => {
            let enabled = enabled_routes(group)?;
            match declared_strategy(group) {
                None => {
                    let (client, request) = compile_route(&enabled[0], &group.request_config)?;
                    Ok(CompiledModelSelection {
                        model_client_config: client,
                        model_request_config: request,
                        selected_type: SelectionKind::ModelGroup,
                        selected_id: group.group_id.clone(),
                    })
                }
                Some(raw) => {
                    let (client, request) = compile_routed_group(group, &enabled, raw)?;
                    Ok(CompiledModelSelection {
                        model_client_config: client,
                        model_request_config: request,
                        selected_type: SelectionKind::ModelGroup,
                        selected_id: group.group_id.clone(),
                    })
                }
            }
        }
    }
}

/// Compiles one route of a group into a concrete config pair.
///
/// The route's request defaults are merged with `group_request_config` and its own
/// `request_overrides`; the client config comes from the route's model. A
/// non-`None` [`ResolvedRoute::timeout`] replaces the client's request timeout.
///
/// # Errors
///
/// Returns [`Error::Validation`] when the route's model has a blank `model_name` or
/// `provider`, when a client option has the wrong type, or when
/// [`ResolvedRoute::timeout`] is not positive.
pub fn compile_route(
    route: &ResolvedRoute,
    group_request_config: &Value,
) -> Result<(ModelClientConfig, ModelRequestConfig)> {
    let mut client = client_config_for(&route.model)?;
    if let Some(seconds) = route.timeout {
        client.timeout = Some(positive_seconds(
            seconds,
            &format!("route {} timeout", route.route_id),
        )?);
    }
    let defaults = merge_request_defaults(&[
        &route.model.request_defaults,
        group_request_config,
        &route.request_overrides,
    ]);
    let request = request_config_for(&route.model.model_name, &defaults);
    Ok((client, request))
}

/// The enabled routes of a group, in routing order.
///
/// Disabled routes are dropped. The result keeps the declared list order **unless**
/// at least one enabled route declares a [`priority`](ResolvedRoute::priority):
/// then the routes are sorted stably by descending priority, treating an absent
/// priority as `0`. Equal priorities therefore keep their declared order, and a
/// group where no route declares a priority is returned untouched (Python
/// behaviour: failover order *is* the declared order).
#[must_use]
pub fn resolve_model_group_routes(group: &ResolvedModelGroup) -> Vec<ResolvedRoute> {
    let mut routes: Vec<ResolvedRoute> = group
        .routes
        .iter()
        .filter(|route| route.enabled)
        .cloned()
        .collect();
    if routes.iter().any(|route| route.priority.is_some()) {
        routes.sort_by_key(|route| std::cmp::Reverse(route.priority.unwrap_or(0)));
    }
    routes
}

/// Merges request defaults from several sources, lowest precedence first.
///
/// Each source is filtered before merging:
///
/// - `null` values are dropped, so "unset" stays unset and the compiler never
///   invents a provider default;
/// - `model`/`model_name` are dropped, because the model name always comes from the
///   model record;
/// - client-configuration keys (`api_key`, `api_base`, `provider`, `client_id`,
///   `endpoint_profile`, `verify_ssl`, the three timeouts, `max_retries`, `headers`)
///   and compiler-owned keys (`deployment`, `messages`, `stream`) are dropped too.
///
/// Nested objects merge recursively and a `null` in a later source never removes a
/// value an earlier source set. Non-object sources contribute nothing.
#[must_use]
pub fn merge_request_defaults(sources: &[&Value]) -> Value {
    let mut merged = Value::Object(Map::new());
    for source in sources {
        let filtered = Value::Object(filter_request_defaults(source));
        merged = deep_merge_values(&merged, &filtered);
    }
    merged
}

/// Deep-merges two JSON values.
///
/// Objects merge key by key and recursively; `null` on the right never removes a
/// value from the left ("`None` cannot unset"); anything else replaces the left
/// value.
#[must_use]
pub(crate) fn deep_merge_values(left: &Value, right: &Value) -> Value {
    match (left, right) {
        (Value::Object(left), Value::Object(right)) => Value::Object(deep_merge_maps(left, right)),
        (_, Value::Null) => left.clone(),
        (_, right) => right.clone(),
    }
}

/// Recursive object merge behind [`deep_merge_values`].
fn deep_merge_maps(left: &Map<String, Value>, right: &Map<String, Value>) -> Map<String, Value> {
    let mut merged = left.clone();
    for (key, value) in right {
        if value.is_null() {
            continue;
        }
        let next = match (merged.get(key), value) {
            (Some(Value::Object(current)), Value::Object(incoming)) => {
                Value::Object(deep_merge_maps(current, incoming))
            }
            _ => value.clone(),
        };
        merged.insert(key.clone(), next);
    }
    merged
}

/// Filters one raw request-defaults layer down to request-owned keys.
fn filter_request_defaults(source: &Value) -> Map<String, Value> {
    let mut filtered = Map::new();
    let Value::Object(source) = source else {
        return filtered;
    };
    for (key, value) in source {
        if value.is_null() || NON_REQUEST_KEYS.contains(&key.as_str()) {
            continue;
        }
        filtered.insert(key.clone(), value.clone());
    }
    filtered
}

/// Builds the client config of one resolved model.
///
/// `provider`/`api_key`/`api_base` come from the dedicated fields (a blank value
/// means "unset", so the provider's own defaults and the environment apply), and
/// `client_options` fills the rest: `client_id`, `verify_ssl`, `timeout`,
/// `stream_first_chunk_timeout`, `stream_idle_timeout`, `max_retries`, and
/// `headers` map onto the matching fields, while every other key — including
/// `endpoint_profile`, `api_mode`, `intelli_router_provider`, and vendor knobs —
/// is preserved in [`ModelClientConfig::extra`].
pub(crate) fn client_config_for(model: &ResolvedModel) -> Result<ModelClientConfig> {
    validate_model(model)?;
    let mut client = ModelClientConfig::new(model.provider.trim());
    if let Some(api_key) = model
        .api_key
        .as_ref()
        .filter(|key| !key.expose_secret().trim().is_empty())
    {
        client.api_key = Some(api_key.clone());
    }
    if let Some(api_base) = model
        .api_base
        .as_deref()
        .filter(|base| !base.trim().is_empty())
    {
        client.api_base = Some(api_base.to_owned());
    }
    let mut extra = Map::new();
    if let Some(profile) = model
        .endpoint_profile
        .as_deref()
        .filter(|profile| !profile.trim().is_empty())
    {
        extra.insert(
            "endpoint_profile".to_owned(),
            Value::String(profile.to_owned()),
        );
    }
    if let Value::Object(options) = &model.client_options {
        for (key, value) in options {
            if value.is_null() {
                continue;
            }
            let consumed = apply_client_option(&mut client, key, value)?;
            if !consumed && !extra.contains_key(key) {
                extra.insert(key.clone(), value.clone());
            }
        }
    }
    client.extra = Value::Object(extra);
    Ok(client)
}

/// Applies one `client_options` entry onto a client config.
///
/// Returns `true` when the key belonged to a typed field, `false` when the caller
/// should keep the value in [`ModelClientConfig::extra`]. A recognized key with an
/// unexpected value type is kept verbatim rather than silently dropped.
fn apply_client_option(client: &mut ModelClientConfig, key: &str, value: &Value) -> Result<bool> {
    match key {
        // The explicit fields win; these options are transport-owned aliases.
        "api_key" | "api_base" | "client_provider" | "provider" => Ok(true),
        "client_id" => match value.as_str().map(str::trim).filter(|id| !id.is_empty()) {
            Some(id) => {
                id.clone_into(&mut client.client_id);
                Ok(true)
            }
            None => Ok(false),
        },
        "verify_ssl" => match value.as_bool() {
            Some(flag) => {
                client.verify_ssl = flag;
                Ok(true)
            }
            None => Ok(false),
        },
        "timeout" => match duration_option(key, value)? {
            Some(duration) => {
                client.timeout = Some(duration);
                Ok(true)
            }
            None => Ok(false),
        },
        "stream_first_chunk_timeout" => match duration_option(key, value)? {
            Some(duration) => {
                client.stream_first_chunk_timeout = Some(duration);
                Ok(true)
            }
            None => Ok(false),
        },
        "stream_idle_timeout" => match duration_option(key, value)? {
            Some(duration) => {
                client.stream_idle_timeout = Some(duration);
                Ok(true)
            }
            None => Ok(false),
        },
        "max_retries" => match value.as_u64() {
            Some(retries) => {
                client.max_retries = u32::try_from(retries).unwrap_or(u32::MAX);
                Ok(true)
            }
            None => Ok(false),
        },
        "headers" => match value {
            Value::Object(headers) => {
                for (name, header) in headers {
                    if let Some(header) = header.as_str() {
                        client.headers.insert(name.clone(), header.to_owned());
                    }
                }
                Ok(true)
            }
            _ => Ok(false),
        },
        _ => Ok(false),
    }
}

/// Builds the request config of one route from already-merged defaults.
///
/// `temperature`, `top_p`, `max_tokens`, `stop`, `tools`, and `tool_choice` map onto
/// the typed fields; a `"extra"` object is flattened into
/// [`ModelRequestConfig::extra`] together with every unrecognized key, so vendor
/// knobs (including a `reasoning` intent) survive untouched. `context_window` is
/// dropped: it is core-side metadata the Rust request config does not model.
pub(crate) fn request_config_for(model_name: &str, defaults: &Value) -> ModelRequestConfig {
    let mut request = ModelRequestConfig::new(model_name);
    let mut extra = Map::new();
    if let Value::Object(defaults) = defaults {
        for (key, value) in defaults {
            if value.is_null() {
                continue;
            }
            let consumed = match key.as_str() {
                "temperature" => match value.as_f64() {
                    Some(temperature) => {
                        request.temperature = Some(temperature as f32);
                        true
                    }
                    None => false,
                },
                "top_p" => match value.as_f64() {
                    Some(top_p) => {
                        request.top_p = Some(top_p as f32);
                        true
                    }
                    None => false,
                },
                "max_tokens" => match value.as_u64() {
                    Some(max_tokens) => {
                        request.max_tokens = Some(u32::try_from(max_tokens).unwrap_or(u32::MAX));
                        true
                    }
                    None => false,
                },
                "stop" => match stop_sequences(value) {
                    Some(stop) => {
                        request.stop = stop;
                        true
                    }
                    None => false,
                },
                "tools" => {
                    request.tools = Some(value.clone());
                    true
                }
                "tool_choice" => {
                    request.tool_choice = Some(value.clone());
                    true
                }
                "context_window" => true,
                "extra" => match value {
                    Value::Object(inner) => {
                        for (inner_key, inner_value) in inner {
                            extra.insert(inner_key.clone(), inner_value.clone());
                        }
                        true
                    }
                    _ => false,
                },
                _ => false,
            };
            if !consumed {
                extra.insert(key.clone(), value.clone());
            }
        }
    }
    request.extra = if extra.is_empty() {
        Value::Null
    } else {
        Value::Object(extra)
    };
    request
}

/// Reads a `stop` setting in either accepted form: one string or an array.
fn stop_sequences(value: &Value) -> Option<Vec<String>> {
    match value {
        Value::String(stop) => Some(vec![stop.clone()]),
        Value::Array(items) => items
            .iter()
            .map(|item| item.as_str().map(str::to_owned))
            .collect(),
        _ => None,
    }
}

/// The strategy a group declares, or `None` when it does not route.
fn declared_strategy(group: &ResolvedModelGroup) -> Option<&str> {
    group
        .strategy
        .as_deref()
        .map(str::trim)
        .filter(|raw| !raw.is_empty())
}

/// The enabled routes of a group, or a validation error naming the group.
pub(crate) fn enabled_routes(group: &ResolvedModelGroup) -> Result<Vec<ResolvedRoute>> {
    let routes = resolve_model_group_routes(group);
    if routes.is_empty() {
        return Err(Error::validation(format!(
            "model_group_id={} has no enabled routes",
            group.group_id
        )));
    }
    Ok(routes)
}

/// Validates a model's identity: `model_name` and `provider` must be non-blank.
fn validate_model(model: &ResolvedModel) -> Result<()> {
    if model.model_name.trim().is_empty() {
        return Err(Error::validation(format!(
            "model_id={} has empty model_name",
            model.model_id
        )));
    }
    if model.provider.trim().is_empty() {
        return Err(Error::validation(format!(
            "model_id={} has empty provider",
            model.model_id
        )));
    }
    Ok(())
}

/// Compiles a group that declares a routing strategy.
fn compile_routed_group(
    group: &ResolvedModelGroup,
    enabled: &[ResolvedRoute],
    raw_strategy: &str,
) -> Result<(ModelClientConfig, ModelRequestConfig)> {
    let strategy = RoutingStrategy::parse(raw_strategy);
    if !strategy.is_known() {
        return Err(Error::validation(format!(
            "unknown routing strategy {:?} for model_group_id={}; supported strategies: {}, {}",
            strategy.as_str(),
            group.group_id,
            RoutingStrategy::ORDERED_FAILOVER,
            RoutingStrategy::TAG_FILTERED
        )));
    }
    // A `tag-filtered` group must name at least one tag when it configures one at
    // all; the compiler validates the argument, the routing model filters by it.
    strategy_tags(
        &group.strategy_kwargs,
        group.fallback_tag.as_deref(),
        &group.group_id,
    )?;
    let num_retries = group
        .num_retries
        .unwrap_or_else(|| u32::try_from(enabled.len().saturating_sub(1)).unwrap_or(u32::MAX));
    let mut plan = Map::new();
    plan.insert(
        "model_group_id".to_owned(),
        Value::String(group.group_id.clone()),
    );
    plan.insert(
        "route_ids".to_owned(),
        Value::Array(
            enabled
                .iter()
                .map(|route| Value::String(route.route_id.clone()))
                .collect(),
        ),
    );
    plan.insert(
        "strategy".to_owned(),
        Value::String(strategy.as_str().to_owned()),
    );
    plan.insert("num_retries".to_owned(), Value::from(num_retries));
    if let Value::Object(kwargs) = &group.strategy_kwargs {
        if !kwargs.is_empty() {
            plan.insert("strategy_kwargs".to_owned(), group.strategy_kwargs.clone());
        }
    }
    if let Some(timeout) = group.timeout_seconds {
        plan.insert("timeout_seconds".to_owned(), Value::from(timeout));
    }
    if let Some(tag) = &group.fallback_tag {
        plan.insert("fallback_tag".to_owned(), Value::String(tag.clone()));
    }
    let mut extra = Map::new();
    extra.insert(ROUTING_PLAN_KEY.to_owned(), Value::Object(plan));
    let client = ModelClientConfig::new(INTELLI_ROUTER_PROVIDER).with_extra(Value::Object(extra));
    let mut request = ModelRequestConfig::new(group.group_id.clone());
    if let Some(context_window) = min_context_window(enabled, &group.request_config) {
        let mut request_extra = Map::new();
        request_extra.insert("context_window".to_owned(), Value::from(context_window));
        request.extra = Value::Object(request_extra);
    }
    Ok((client, request))
}

/// The tag(s) a `tag-filtered` strategy pins routes to.
///
/// `strategy_kwargs.tag` (or the Python spelling `fallback_tag`) accepts one string
/// or an array of strings; when neither is configured the group-level
/// [`ResolvedModelGroup::fallback_tag`] applies, and when that is unset too the
/// result is empty and no filtering happens.
///
/// # Errors
///
/// Returns [`Error::Validation`] when the argument is neither a string nor an array
/// of strings.
pub(crate) fn strategy_tags(
    strategy_kwargs: &Value,
    fallback_tag: Option<&str>,
    model_group_id: &str,
) -> Result<Vec<String>> {
    let raw = strategy_kwargs
        .get("tag")
        .or_else(|| strategy_kwargs.get("fallback_tag"));
    match raw {
        None | Some(Value::Null) => Ok(fallback_tag
            .map(|tag| vec![tag.to_owned()])
            .unwrap_or_default()),
        Some(Value::String(tag)) => Ok(vec![tag.clone()]),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| match item.as_str() {
                Some(tag) => Ok(tag.to_owned()),
                None => Err(Error::validation(format!(
                    "model_group_id={model_group_id}: routing.strategy_kwargs.tag entries \
                     must be strings, found {}",
                    json_kind(item)
                ))),
            })
            .collect(),
        Some(other) => Err(Error::validation(format!(
            "model_group_id={model_group_id}: routing.strategy_kwargs.tag must be a string \
             or an array of strings, found {}",
            json_kind(other)
        ))),
    }
}

/// The smallest positive `context_window` across the enabled routes, if any.
fn min_context_window(routes: &[ResolvedRoute], group_request_config: &Value) -> Option<u64> {
    routes
        .iter()
        .filter_map(|route| {
            let defaults = merge_request_defaults(&[
                &route.model.request_defaults,
                group_request_config,
                &route.request_overrides,
            ]);
            defaults.get("context_window").and_then(Value::as_u64)
        })
        .filter(|window| *window > 0)
        .min()
}

/// Reads a seconds-valued client option as a [`Duration`].
///
/// Returns `Ok(None)` when the value is not a number (the caller keeps it in
/// `extra` instead of dropping it) and a validation error when it is not positive.
fn duration_option(key: &str, value: &Value) -> Result<Option<Duration>> {
    let Some(seconds) = value.as_f64() else {
        return Ok(None);
    };
    positive_seconds(seconds, &format!("client_options.{key}")).map(Some)
}

/// Converts a positive number of seconds into a [`Duration`].
fn positive_seconds(seconds: f64, context: &str) -> Result<Duration> {
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(Error::validation(format!(
            "{context} must be a positive number of seconds, found {seconds}"
        )));
    }
    Ok(Duration::from_secs_f64(seconds))
}

/// Compiles a single resolved model.
fn compile_model(model: &ResolvedModel) -> Result<(ModelClientConfig, ModelRequestConfig)> {
    let client = client_config_for(model)?;
    let defaults = merge_request_defaults(&[&model.request_defaults]);
    let request = request_config_for(&model.model_name, &defaults);
    Ok((client, request))
}
