//! Built-in `branch` workflow component.
//!
//! Ports Python's `openjiuwen.core.workflow.components.flow.branch_comp.BranchComponent`
//! and its `BranchRouter`. A branch node is a routing decision, not a data transform:
//! [`BranchComponent::invoke`] returns `{}` and the engine asks
//! [`BranchComponent::route`] which component ids to activate next.
//!
//! Two seams expose the same [`BranchRouter`]:
//!
//! - [`Component::route`] for the workflow engine's per-component routing hook.
//! - [`BranchComponent::router_adapter`], an [`openjiuwen_graph::Router`] that reads the
//!   graph state as the router source. `RouterContext` has no public constructor, so the
//!   adapter is exercised by the engine's integration tests rather than its unit tests.

use std::sync::Arc;

use serde_json::{json, Value};

use openjiuwen_common::error::Result;

use crate::ability::ComponentAbility;
use crate::component::{Component, ComponentCard};
use crate::condition::BranchRouter;
use crate::context::ComponentContext;

/// The `branch` built-in component.
///
/// The component owns an [`Arc<BranchRouter>`] so its conditions can be shared with the
/// graph router adapter without cloning the parsed conditions.
#[derive(Debug, Clone)]
pub struct BranchComponent {
    router: Arc<BranchRouter>,
}

impl BranchComponent {
    /// Creates a branch component that evaluates `router`.
    #[must_use]
    pub fn new(router: BranchRouter) -> Self {
        Self {
            router: Arc::new(router),
        }
    }

    /// The router this component evaluates.
    #[must_use]
    pub fn router(&self) -> &Arc<BranchRouter> {
        &self.router
    }

    /// A graph-engine router over the same [`BranchRouter`].
    #[must_use]
    pub fn router_adapter(&self) -> Arc<dyn openjiuwen_graph::Router> {
        Arc::new(BranchRouterAdapter {
            router: Arc::clone(&self.router),
        })
    }
}

#[async_trait::async_trait]
impl Component for BranchComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("branch", "branch").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &'static str {
        "branch"
    }

    /// A branch produces no data of its own; it only selects successors.
    async fn invoke(&self, _inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        Ok(json!({}))
    }

    fn route(&self, source: &Value) -> Option<Result<Vec<String>>> {
        Some(self.router.route(source))
    }
}

/// [`openjiuwen_graph::Router`] adapter over a [`BranchRouter`].
///
/// The adapter routes from `RouterContext::state()`, which is the graph state after the
/// branch node's (empty) result was merged, so `${path}` conditions see the same data the
/// engine sees.
#[derive(Debug, Clone)]
pub struct BranchRouterAdapter {
    router: Arc<BranchRouter>,
}

impl BranchRouterAdapter {
    /// Wraps `router`.
    #[must_use]
    pub fn new(router: Arc<BranchRouter>) -> Self {
        Self { router }
    }

    /// The wrapped router.
    #[must_use]
    pub fn router(&self) -> &Arc<BranchRouter> {
        &self.router
    }
}

#[async_trait::async_trait]
impl openjiuwen_graph::Router for BranchRouterAdapter {
    fn name(&self) -> &'static str {
        "branch"
    }

    async fn route(&self, ctx: &openjiuwen_graph::RouterContext) -> Result<Vec<String>> {
        self.router.route(&ctx.state().as_value())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use std::sync::Arc;

    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_graph::Router as _;
    use openjiuwen_session::{Session, WorkflowSession};
    use serde_json::json;

    use super::*;
    use crate::condition::{Branch, Condition};

    fn context() -> ComponentContext {
        let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("branch-tests"));
        ComponentContext::new(session, "branch", "wf")
    }

    fn router() -> BranchRouter {
        let mut router = BranchRouter::new();
        let _ = router
            .add_branch(Branch::new(
                Condition::new("${score} > 10").unwrap(),
                vec!["high".to_owned()],
            ))
            .add_branch(Branch::new(
                Condition::new("${score} > 0").unwrap(),
                vec!["low".to_owned()],
            ));
        router
    }

    #[test]
    fn route_delegates_and_preserves_first_match() {
        let component = BranchComponent::new(router());
        // Both conditions hold for `score = 20`; the first branch wins.
        assert_eq!(
            component.route(&json!({"score": 20})).unwrap().unwrap(),
            vec!["high".to_owned()]
        );
        assert_eq!(
            component.route(&json!({"score": 5})).unwrap().unwrap(),
            vec!["low".to_owned()]
        );
    }

    #[test]
    fn route_reports_the_python_no_match_error() {
        let error = BranchComponent::new(router())
            .route(&json!({"score": 0}))
            .unwrap()
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error.message().contains("101021"));
    }

    #[tokio::test]
    async fn invoke_returns_an_empty_object() {
        let component = BranchComponent::new(router());
        let output = component
            .invoke(json!({"score": 20}), &context())
            .await
            .unwrap();
        assert_eq!(output, json!({}));
    }

    #[test]
    fn identity_matches_python() {
        let component = BranchComponent::new(BranchRouter::new());
        assert_eq!(component.component_type(), "branch");
        assert_eq!(component.abilities(), vec![ComponentAbility::Invoke]);
        let card = component.card();
        assert_eq!(card.id, "branch");
        assert_eq!(card.component_type, "branch");
        assert_eq!(card.abilities, vec![ComponentAbility::Invoke]);
    }

    #[test]
    fn adapter_names_itself_and_shares_the_router() {
        let router = router();
        let targets = router.targets();
        let component = BranchComponent::new(router);
        let adapter = component.router_adapter();
        assert_eq!(adapter.name(), "branch");
        // The adapter and the component observe the same router instance.
        let explicit = BranchRouterAdapter::new(Arc::clone(component.router()));
        assert_eq!(explicit.name(), "branch");
        assert_eq!(explicit.router().targets(), targets);
        assert!(Arc::ptr_eq(explicit.router(), component.router()));
    }
}
