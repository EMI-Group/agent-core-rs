//! IntelliRouter-style model routing: selection compilation, failover, tag pinning.
//!
//! Ports `openjiuwen/core/foundation/llm/routing/` (schema + compiler) and replaces
//! the optional `intelli_router` vendor package with an in-crate routing model.
//! Nothing here reads a database, a configuration file, or the network while
//! compiling; resolving a selection id into routes stays in the calling layer.
//!
//! Three layers, each independently usable:
//!
//! 1. **Schema** — [`ResolvedModel`], [`ResolvedRoute`], [`ResolvedModelGroup`],
//!    [`ResolvedRoutingConfig`], and [`CompiledModelSelection`] are plain
//!    `serde` data. A persisted [`ModelSelection`] is the reference an upper layer
//!    resolves into one of them.
//! 2. **Compiler** — [`compile_model_selection`] turns a resolved payload into the
//!    `ModelClientConfig` / `ModelRequestConfig` pair a model consumes;
//!    [`compile_route`] does the same for a single route, and
//!    [`resolve_model_group_routes`] applies the additive priority rule layer.
//! 3. **Routing** — [`RoutingModel`] implements
//!    [`Model`](crate::llm::model::Model) over the enabled routes of a group:
//!    ordered failover, tag pinning, and streaming that only fails over before the
//!    first chunk reaches the caller.
//!
//! ```
//! use openjiuwen_foundation::llm::routing::{
//!     compile_model_selection, ModelSelectionWire, ResolvedModel,
//! };
//! use serde_json::json;
//!
//! let wire = ModelSelectionWire::from_value(json!({
//!     "model_id": "m-1",
//!     "model_name": "gpt-4o-mini",
//!     "provider": "openai",
//!     "api_base": "https://api.example/v1",
//!     "request_defaults": {"temperature": 0.2}
//! }))
//! .expect("a resolved model payload");
//!
//! let compiled = compile_model_selection(&wire).expect("the payload compiles");
//! assert_eq!(compiled.model_client_config.provider, "openai");
//! assert_eq!(compiled.model_request_config.model_name, "gpt-4o-mini");
//! assert_eq!(compiled.model_request_config.temperature, Some(0.2));
//! ```
//!
//! Documented gaps (see the crate `docs/PARITY.md` for the parity matrix):
//!
//! - No TPM/RPM throttling: [`ResolvedRoute::tpm`] and [`ResolvedRoute::rpm`] are
//!   recorded but not enforced, and there is no cross-request circuit breaker or
//!   cooldown state;
//! - No tag-based cost/latency scoring: `tag-filtered` is a filter, not a scorer;
//! - An unknown strategy name is rejected by [`compile_model_selection`], while
//!   [`RoutingModel`] treats a [`RoutingStrategy::Custom`] name as
//!   `ordered-failover`;
//! - The group `timeout_seconds` is recorded in the compiled plan but not enforced.
pub mod compiler;
pub mod model;
pub mod schema;

pub use compiler::{
    compile_model_selection, compile_route, merge_request_defaults, resolve_model_group_routes,
    INTELLI_ROUTER_PROVIDER, ROUTING_PLAN_KEY,
};
pub use model::{is_retryable_failure, RoutingModel};
pub use schema::{
    CompiledModelSelection, ModelSelection, ModelSelectionWire, ResolvedModel, ResolvedModelGroup,
    ResolvedRoute, ResolvedRoutingConfig, RoutingStrategy, SelectionKind,
};

#[cfg(test)]
mod tests;
