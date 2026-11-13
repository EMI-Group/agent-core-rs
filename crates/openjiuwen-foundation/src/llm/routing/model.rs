//! The routing model: one [`Model`] backed by an ordered set of route models.
//!
//! This is the Rust replacement for the Python `intelli_router` package usage in
//! `openjiuwen/core/foundation/llm/model_clients/intelli_router_model_client.py`:
//! instead of delegating to a vendor router, [`RoutingModel`] builds one real model
//! client per enabled route and implements the failover policy itself.
//!
//! Behaviour, precisely:
//!
//! - **Candidates** — the enabled routes of the group in routing order (declared
//!   order, or stable descending [`priority`](crate::llm::routing::schema::ResolvedRoute::priority)
//!   when any route declares one; see
//!   [`resolve_model_group_routes`](super::compiler::resolve_model_group_routes)).
//! - **Attempt budget** — `num_retries + 1` attempts, capped at the number of
//!   candidates: the router makes one pass over the candidates and never cycles.
//!   `num_retries` defaults to `candidates - 1`, so a default group tries every
//!   candidate once.
//! - **Retryable failures** — [`Error::Transport`], [`Error::Timeout`],
//!   [`Error::Llm`], and [`Error::External`] fall over to the next candidate.
//!   [`Error::Validation`], [`Error::Config`], [`Error::NotFound`],
//!   [`Error::Unsupported`], and every other kind return immediately: a bad
//!   request, a missing credential, or a missing endpoint will not fix itself on
//!   another route.
//! - **Streaming** — the same failover, but only until the first chunk reaches the
//!   caller. An error before the first chunk falls over to the next candidate; an
//!   error after a chunk was emitted is propagated, because the caller already saw
//!   partial output and a silent switch would splice two answers together.
//! - **Tag filtering** — the `tag-filtered` strategy keeps only candidates tagged
//!   with `strategy_kwargs.tag` (string or array); an empty filtered set is a
//!   [`Error::Validation`]. [`RoutingStrategy::Custom`] names behave as
//!   `ordered-failover`.
//!
//! Every per-route call uses the route's compiled request config as defaults and
//! lets the caller's request win where it is set (including a deep merge of
//! [`ModelRequestConfig::extra`]).
use std::fmt;
use std::sync::Arc;

use async_stream::try_stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use tracing::{debug, warn};

use super::compiler::{compile_route, enabled_routes, strategy_tags, INTELLI_ROUTER_PROVIDER};
use super::schema::{ResolvedModelGroup, ResolvedRoute, ResolvedRoutingConfig, RoutingStrategy};
use crate::llm::config::ModelRequestConfig;
use crate::llm::factory::init_model;
use crate::llm::model::{ChatResponse, Model, StreamChunk};
use crate::prompt::BaseMessage;

/// Tracing target of the routing layer.
const TRACING_TARGET: &str = "openjiuwen_foundation::llm::routing";

/// Whether a failed route attempt may be retried on the next candidate.
///
/// Transport problems, deadlines, provider failures, and wrapped foreign errors are
/// considered transient. Everything else — a validation error, a configuration or
/// credential problem, a missing model, an unsupported provider — is returned to
/// the caller unchanged, because another route cannot fix it.
#[must_use]
pub fn is_retryable_failure(error: &Error) -> bool {
    matches!(
        error.kind(),
        ErrorKind::Transport | ErrorKind::Timeout | ErrorKind::Llm | ErrorKind::External
    )
}

/// A [`Model`] that routes a call over the enabled routes of a model group.
///
/// Construct one with [`RoutingModel::from_group`], which builds a real model client
/// per route, or with [`RoutingModel::with_models`] when the caller already owns the
/// per-route models. The identity it reports is the routing identity:
/// [`provider`](Model::provider) is [`INTELLI_ROUTER_PROVIDER`] unless overridden
/// with [`with_provider`](Self::with_provider), and
/// [`model_name`](Model::model_name) is the group id.
pub struct RoutingModel {
    provider: String,
    model_group_id: String,
    entries: Vec<RoutingEntry>,
    routing: ResolvedRoutingConfig,
    fallback_tag: Option<String>,
}

/// One candidate: its resolved route, the built model, and its request defaults.
struct RoutingEntry {
    route: ResolvedRoute,
    model: Arc<dyn Model>,
    request: ModelRequestConfig,
}

impl RoutingEntry {
    /// Builds the request one route call uses.
    ///
    /// The route's compiled request config supplies the defaults (model name,
    /// sampling settings, vendor knobs); the caller's request wins wherever it sets
    /// a value, and both `extra` maps are deep-merged with the caller's keys on top.
    fn effective_request(&self, caller: &ModelRequestConfig) -> ModelRequestConfig {
        let base = &self.request;
        let model_name = if base.model_name.trim().is_empty() {
            self.route.model.model_name.clone()
        } else {
            base.model_name.clone()
        };
        ModelRequestConfig {
            model_name,
            temperature: caller.temperature.or(base.temperature),
            top_p: caller.top_p.or(base.top_p),
            max_tokens: caller.max_tokens.or(base.max_tokens),
            stop: if caller.stop.is_empty() {
                base.stop.clone()
            } else {
                caller.stop.clone()
            },
            context_window: caller.context_window.or(base.context_window),
            reasoning: caller.reasoning.clone().or_else(|| base.reasoning.clone()),
            tools: caller.tools.clone().or_else(|| base.tools.clone()),
            tool_choice: caller
                .tool_choice
                .clone()
                .or_else(|| base.tool_choice.clone()),
            stream: caller.stream,
            extra: super::compiler::deep_merge_values(&base.extra, &caller.extra),
        }
    }

    /// Whether this route carries any of `tags` (as model `fallback_tag` or route `tag`).
    fn matches_any(&self, tags: &[String]) -> bool {
        tags.iter().any(|tag| {
            self.route.model.fallback_tag.as_deref() == Some(tag.as_str())
                || self.route.tag.as_deref() == Some(tag.as_str())
        })
    }
}

/// A route prepared for one streamed call; owned so the stream is `'static`.
struct PreparedRoute {
    route_id: String,
    model_name: String,
    model: Arc<dyn Model>,
    request: ModelRequestConfig,
}

impl RoutingModel {
    /// Builds a routing model from a resolved model group.
    ///
    /// Every enabled route is compiled (request defaults merged) and realised through
    /// [`init_model`] eagerly, so a configuration problem surfaces here rather than
    /// mid-conversation. Routes are taken in routing order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the group has no enabled route or a route
    /// has a blank model name/provider or a malformed client option, and whatever
    /// [`init_model`] returns for a route (typically [`Error::Config`] for a missing
    /// endpoint or credential, [`Error::Unsupported`] for an unknown provider).
    pub async fn from_group(group: &ResolvedModelGroup) -> Result<Self> {
        let routes = enabled_routes(group)?;
        let mut entries = Vec::with_capacity(routes.len());
        for route in routes {
            let (client, request) = compile_route(&route, &group.request_config)?;
            let model = init_model(client, request.clone()).await?;
            entries.push(RoutingEntry {
                route,
                model,
                request,
            });
        }
        Ok(Self::assemble(group, entries))
    }

    /// Builds a routing model from caller-supplied route models.
    ///
    /// `models` must be given in routing order and match the number of enabled
    /// routes one for one; this is the seam for tests and for callers that
    /// construct clients themselves.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the group has no enabled route, when
    /// `models` does not have exactly one entry per enabled route, or when a route's
    /// config cannot be compiled.
    pub fn with_models(group: &ResolvedModelGroup, models: Vec<Arc<dyn Model>>) -> Result<Self> {
        let routes = enabled_routes(group)?;
        if routes.len() != models.len() {
            return Err(Error::validation(format!(
                "model_group_id={} has {} enabled route(s) but {} model(s) were supplied",
                group.group_id,
                routes.len(),
                models.len()
            )));
        }
        let mut entries = Vec::with_capacity(routes.len());
        for (route, model) in routes.into_iter().zip(models) {
            let (_, request) = compile_route(&route, &group.request_config)?;
            entries.push(RoutingEntry {
                route,
                model,
                request,
            });
        }
        Ok(Self::assemble(group, entries))
    }

    /// Overrides the provider name this routing model reports.
    #[must_use]
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }

    /// The id of the model group this model routes over.
    #[must_use]
    pub fn model_group_id(&self) -> &str {
        &self.model_group_id
    }

    /// The routing policy with defaults resolved.
    #[must_use]
    pub fn routing_config(&self) -> &ResolvedRoutingConfig {
        &self.routing
    }

    /// The resolved enabled routes, in routing order (introspection).
    #[must_use]
    pub fn routes(&self) -> Vec<&ResolvedRoute> {
        self.entries.iter().map(|entry| &entry.route).collect()
    }

    /// The ids of the resolved enabled routes, in routing order.
    #[must_use]
    pub fn route_ids(&self) -> Vec<&str> {
        self.entries
            .iter()
            .map(|entry| entry.route.route_id.as_str())
            .collect()
    }

    /// How many enabled routes this model can use.
    #[must_use]
    pub fn route_count(&self) -> usize {
        self.entries.len()
    }

    /// Assembles the model from already-built entries.
    fn assemble(group: &ResolvedModelGroup, entries: Vec<RoutingEntry>) -> Self {
        Self {
            provider: INTELLI_ROUTER_PROVIDER.to_owned(),
            model_group_id: group.group_id.clone(),
            entries,
            routing: group.routing(),
            fallback_tag: group.fallback_tag.clone(),
        }
    }

    /// The candidates of one call, after tag filtering.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the tag argument is malformed or when a
    /// `tag-filtered` strategy leaves no candidate.
    fn candidates(&self) -> Result<Vec<&RoutingEntry>> {
        if self.routing.strategy != RoutingStrategy::TagFiltered {
            return Ok(self.entries.iter().collect());
        }
        let tags = strategy_tags(
            &self.routing.strategy_kwargs,
            self.fallback_tag.as_deref(),
            &self.model_group_id,
        )?;
        if tags.is_empty() {
            return Ok(self.entries.iter().collect());
        }
        let filtered: Vec<&RoutingEntry> = self
            .entries
            .iter()
            .filter(|entry| entry.matches_any(&tags))
            .collect();
        if filtered.is_empty() {
            return Err(Error::validation(format!(
                "model_group_id={} has no enabled route tagged [{}]; available tags: [{}]",
                self.model_group_id,
                tags.join(", "),
                self.available_tags().join(", ")
            )));
        }
        Ok(filtered)
    }

    /// The distinct tags the enabled routes declare, for error messages.
    fn available_tags(&self) -> Vec<String> {
        let mut tags: Vec<String> = Vec::new();
        for entry in &self.entries {
            for tag in entry
                .route
                .model
                .fallback_tag
                .iter()
                .chain(entry.route.tag.iter())
            {
                if !tags.contains(tag) {
                    tags.push(tag.clone());
                }
            }
        }
        tags
    }

    /// The candidates one call may actually attempt, honouring `num_retries`.
    ///
    /// # Errors
    ///
    /// Propagates the candidate errors of [`candidates`](Self::candidates).
    fn attempts(&self) -> Result<Vec<&RoutingEntry>> {
        let candidates = self.candidates()?;
        let budget = usize::try_from(self.routing.num_retries)
            .unwrap_or(usize::MAX)
            .saturating_add(1)
            .min(candidates.len())
            .max(1);
        Ok(candidates.into_iter().take(budget).collect())
    }

    /// The error returned when every attempted route failed.
    fn exhausted_error(&self, attempted: &[String], last: Option<Error>) -> Error {
        exhausted_error(&self.model_group_id, attempted, last)
    }
}

/// Builds the "all routes failed" error, naming the attempted routes and the cause.
fn exhausted_error(model_group_id: &str, attempted: &[String], last: Option<Error>) -> Error {
    let routes = attempted.join(", ");
    match last {
        Some(error) => Error::llm(format!(
            "all routes failed for model_group_id={model_group_id}: attempted=[{routes}]; \
             last: {} ({})",
            error.kind(),
            error.message()
        )),
        None => Error::llm(format!(
            "all routes failed for model_group_id={model_group_id}: attempted=[{routes}]; \
             no route reported a cause"
        )),
    }
}

impl fmt::Debug for RoutingModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RoutingModel")
            .field("provider", &self.provider)
            .field("model_group_id", &self.model_group_id)
            .field("strategy", &self.routing.strategy)
            .field("num_retries", &self.routing.num_retries)
            .field("fallback_tag", &self.fallback_tag)
            .field("route_ids", &self.route_ids())
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Model for RoutingModel {
    fn provider(&self) -> &str {
        &self.provider
    }

    fn model_name(&self) -> &str {
        &self.model_group_id
    }

    async fn invoke(
        &self,
        messages: &[BaseMessage],
        request: &ModelRequestConfig,
    ) -> Result<ChatResponse> {
        let attempts = self.attempts()?;
        let mut attempted: Vec<String> = Vec::with_capacity(attempts.len());
        let mut last_error: Option<Error> = None;
        for entry in attempts {
            attempted.push(entry.route.route_id.clone());
            let response = entry
                .model
                .invoke(messages, &entry.effective_request(request))
                .await;
            match response {
                Ok(response) => {
                    if attempted.len() > 1 {
                        debug!(
                            target: TRACING_TARGET,
                            group = %self.model_group_id,
                            route = %entry.route.route_id,
                            model = %entry.route.model.model_name,
                            attempts = attempted.len(),
                            "model route answered after failover"
                        );
                    }
                    return Ok(response);
                }
                Err(error) => {
                    if !is_retryable_failure(&error) {
                        return Err(error);
                    }
                    warn!(
                        target: TRACING_TARGET,
                        group = %self.model_group_id,
                        route = %entry.route.route_id,
                        model = %entry.route.model.model_name,
                        kind = %error.kind(),
                        "model route failed; falling over to the next route"
                    );
                    last_error = Some(error);
                }
            }
        }
        Err(self.exhausted_error(&attempted, last_error))
    }

    fn stream(
        &self,
        messages: Vec<BaseMessage>,
        request: ModelRequestConfig,
    ) -> BoxStream<'static, Result<StreamChunk>> {
        let prepared: Vec<PreparedRoute> = match self.attempts() {
            Ok(attempts) => attempts
                .into_iter()
                .map(|entry| PreparedRoute {
                    route_id: entry.route.route_id.clone(),
                    model_name: entry.route.model.model_name.clone(),
                    model: Arc::clone(&entry.model),
                    request: entry.effective_request(&request),
                })
                .collect(),
            Err(error) => return Box::pin(futures::stream::once(async move { Err(error) })),
        };
        let group_id = self.model_group_id.clone();
        Box::pin(try_stream! {
            let mut attempted: Vec<String> = Vec::with_capacity(prepared.len());
            let mut last_error: Option<Error> = None;
            let mut emitted = false;
            for route in prepared {
                attempted.push(route.route_id.clone());
                let mut stream = route.model.stream(messages.clone(), route.request);
                let mut failure: Option<Error> = None;
                while let Some(item) = stream.next().await {
                    match item {
                        Ok(chunk) => {
                            emitted = true;
                            yield chunk;
                        }
                        Err(error) => {
                            failure = Some(error);
                            break;
                        }
                    }
                }
                let Some(error) = failure else {
                    return;
                };
                if emitted || !is_retryable_failure(&error) {
                    warn!(
                        target: TRACING_TARGET,
                        group = %group_id,
                        route = %route.route_id,
                        model = %route.model_name,
                        kind = %error.kind(),
                        emitted,
                        "model route stream failed; propagating the failure"
                    );
                    Err::<(), Error>(error)?;
                    return;
                }
                warn!(
                    target: TRACING_TARGET,
                    group = %group_id,
                    route = %route.route_id,
                    model = %route.model_name,
                    kind = %error.kind(),
                    "model route failed before its first chunk; falling over to the next route"
                );
                last_error = Some(error);
            }
            Err::<(), Error>(exhausted_error(&group_id, &attempted, last_error))?;
        })
    }
}
