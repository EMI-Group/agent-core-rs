//! The workflow engine: the recorded definition and its compile-time validation.
//!
//! [`WorkflowDefinition`] is what the builder records: components with their node
//! configuration, edges, conditional fan-out, the start/end ids, and any problems the
//! builder noticed. [`validate::validate`] turns a definition into one coded compile
//! error listing every problem.
//!
//! The graph adapter (`node`) and `CompiledWorkflow` are added by later stages; this
//! module only exposes the definition types and the validation entry point, so the
//! builder can record and check a workflow without compiling a graph.

pub mod definition;
pub mod validate;

pub use definition::{ConditionalSource, Edge, EdgeKind, Entry, NodeConfig, WorkflowDefinition};

/// Fakes shared by the engine's unit tests.
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Arc;

    use openjiuwen_graph::Router;

    use crate::ability::ComponentAbility;
    use crate::card::WorkflowCard;
    use crate::component::{Component, ComponentCard};
    use crate::components::BranchRouterAdapter;
    use crate::condition::BranchRouter;

    /// A component whose type and abilities are fixed by the test.
    #[derive(Clone)]
    struct FakeComponent {
        ty: &'static str,
        abilities: Vec<ComponentAbility>,
    }

    #[async_trait::async_trait]
    impl Component for FakeComponent {
        fn card(&self) -> ComponentCard {
            ComponentCard::new(self.ty, self.ty).with_abilities(self.abilities.clone())
        }

        fn abilities(&self) -> Vec<ComponentAbility> {
            self.abilities.clone()
        }

        fn component_type(&self) -> &str {
            self.ty
        }
    }

    /// A component of type `ty` implementing exactly `abilities`.
    #[must_use]
    pub(crate) fn fake(ty: &'static str, abilities: Vec<ComponentAbility>) -> Arc<dyn Component> {
        Arc::new(FakeComponent { ty, abilities })
    }

    /// A graph router with no branches, enough to register a conditional edge.
    #[must_use]
    pub(crate) fn router() -> Arc<dyn Router> {
        Arc::new(BranchRouterAdapter::new(Arc::new(BranchRouter::new())))
    }

    /// The workflow card every unit test uses.
    #[must_use]
    pub(crate) fn card() -> WorkflowCard {
        WorkflowCard::new("wf")
    }
}
