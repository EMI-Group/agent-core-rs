//! Conditional routing: the [`Router`] trait, router context, and closure
//! adapters.
//!
//! A router runs after a node completes and returns the node ids to activate.
//! An empty result routes nowhere (the branch ends). Every returned id must be a
//! known node, otherwise the run fails with
//! [`openjiuwen_common::error::Error::graph`].

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use openjiuwen_common::error::Result;
use openjiuwen_session::Session;

use crate::state::GraphState;

/// The read-only view a [`Router`] receives.
///
/// The state is shared through an [`Arc`], so every router of one dispatch reads
/// the very same allocation and routing never copies the state.
#[derive(Clone)]
pub struct RouterContext {
    source: String,
    state: Arc<GraphState>,
    superstep: u32,
    session: Arc<dyn Session>,
    metadata: Arc<BTreeMap<String, Value>>,
}

impl RouterContext {
    /// Builds a context over an already shared state; the engine is the only
    /// constructor.
    pub(crate) fn new_shared(
        source: impl Into<String>,
        state: Arc<GraphState>,
        superstep: u32,
        session: Arc<dyn Session>,
        metadata: Arc<BTreeMap<String, Value>>,
    ) -> Self {
        Self {
            source: source.into(),
            state,
            superstep,
            session,
            metadata,
        }
    }

    /// Returns the id of the node the router is attached to.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Returns the state after the source node's result was merged.
    #[must_use]
    pub fn state(&self) -> &GraphState {
        self.state.as_ref()
    }

    /// Returns the shared handle of [`RouterContext::state`].
    ///
    /// Cloning it is free and never copies the state, so a router that needs to
    /// keep the state can retain the handle instead of cloning the map.
    #[must_use]
    pub fn state_arc(&self) -> &Arc<GraphState> {
        &self.state
    }

    /// Returns the superstep the router runs in.
    #[must_use]
    pub fn superstep(&self) -> u32 {
        self.superstep
    }

    /// Returns the run's session.
    #[must_use]
    pub fn session(&self) -> &Arc<dyn Session> {
        &self.session
    }

    /// Returns the graph's metadata.
    #[must_use]
    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.metadata
    }
}

impl std::fmt::Debug for RouterContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RouterContext")
            .field("source", &self.source)
            .field("superstep", &self.superstep)
            .finish_non_exhaustive()
    }
}

/// Chooses the nodes to activate after a source node completes.
#[async_trait]
pub trait Router: Send + Sync {
    /// A human-readable name used in errors and visualization.
    fn name(&self) -> &str;

    /// Returns the target node ids to activate.
    ///
    /// # Errors
    ///
    /// Returns an error when routing cannot be decided; the engine adds the
    /// router name and source node as context.
    async fn route(&self, ctx: &RouterContext) -> Result<Vec<String>>;
}

/// The future a [`FnRouter`] callback returns: boxed so the engine can prove it
/// is `Send`.
pub type BoxRouterFuture<'a> = Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send + 'a>>;

/// A [`Router`] backed by an async closure.
///
/// The callback returns a boxed [`Send`] future (Rust 1.85's `AsyncFn` traits
/// cannot prove to a generic wrapper that their returned future is `Send` on
/// stable, so the boxing is explicit and the adapter stays `Send`).
///
/// ```
/// use openjiuwen_graph::{FnRouter, RouterContext};
///
/// let router = FnRouter::new("either", |ctx: &RouterContext| {
///     Box::pin(async move {
///         let go = ctx.state().get("go").and_then(|v| v.as_bool()).unwrap_or(false);
///         Ok(if go { vec!["a".to_string()] } else { vec!["b".to_string()] })
///     })
/// });
/// ```
pub struct FnRouter<F> {
    name: String,
    f: F,
}

impl<F> FnRouter<F>
where
    F: for<'a> Fn(&'a RouterContext) -> BoxRouterFuture<'a>,
{
    /// Wraps `f` as a router named `name`.
    pub fn new(name: impl Into<String>, f: F) -> Self {
        Self {
            name: name.into(),
            f,
        }
    }
}

impl<F> std::fmt::Debug for FnRouter<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnRouter")
            .field("name", &self.name)
            .field("f", &"<callback>")
            .finish()
    }
}

#[async_trait]
impl<F> Router for FnRouter<F>
where
    F: for<'a> Fn(&'a RouterContext) -> BoxRouterFuture<'a> + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    async fn route(&self, ctx: &RouterContext) -> Result<Vec<String>> {
        (self.f)(ctx).await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use openjiuwen_session::WorkflowSession;
    use serde_json::json;

    #[allow(clippy::unused_async)]
    async fn context() -> RouterContext {
        RouterContext::new_shared(
            "source",
            Arc::new(GraphState::from_value(json!({"go": true})).unwrap()),
            2,
            Arc::new(WorkflowSession::new("s1")) as Arc<dyn Session>,
            Arc::new(BTreeMap::new()),
        )
    }

    #[tokio::test]
    async fn fn_router_routes_from_state() {
        let router = FnRouter::new("either", |ctx: &RouterContext| {
            Box::pin(async move {
                let go = ctx
                    .state()
                    .get("go")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                Ok(if go {
                    vec!["a".to_string()]
                } else {
                    vec!["b".to_string()]
                })
            })
        });
        assert_eq!(router.name(), "either");
        let targets = router.route(&context().await).await.unwrap();
        assert_eq!(targets, vec!["a".to_string()]);
    }

    #[tokio::test]
    async fn fn_router_accepts_async_fn() {
        #[allow(clippy::unused_async)]
        async fn nowhere(_ctx: &RouterContext) -> Result<Vec<String>> {
            Ok(Vec::new())
        }
        let router = FnRouter::new("nowhere", |ctx: &RouterContext| Box::pin(nowhere(ctx)));
        assert!(router.route(&context().await).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn context_accessors() {
        let ctx = context().await;
        assert_eq!(ctx.source(), "source");
        assert_eq!(ctx.superstep(), 2);
        assert_eq!(ctx.state().get("go"), Some(&json!(true)));
        assert_eq!(ctx.session().id(), "s1");
        assert!(ctx.metadata().is_empty());
        assert!(format!("{ctx:?}").contains("source"));
    }

    #[tokio::test]
    async fn context_state_is_shared_not_copied() {
        let ctx = context().await;
        let shared = Arc::clone(ctx.state_arc());
        assert!(Arc::ptr_eq(&shared, ctx.state_arc()));
        assert!(std::ptr::eq(ctx.state(), shared.as_ref()));
        // Cloning a context clones the handle, never the state.
        let cloned = ctx.clone();
        assert!(Arc::ptr_eq(cloned.state_arc(), ctx.state_arc()));
        assert_eq!(Arc::strong_count(&shared), 3);
    }
}
