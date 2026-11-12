//! The self-evolution rail seam.
//!
//! The evolution bodies (RSI, `agent_evolving`, skill-evolution proposals) are
//! deferred, so the harness depends on this small trait instead of on any
//! evolution crate: upper layers implement [`EvolutionRail`], wrap it with
//! [`evolution_adapter`], and register the result like any other rail.
//!
//! The hooks are synchronous by design — evolution work is metadata bookkeeping
//! (proposal drafting, metric sampling) with no model or tool IO — which keeps
//! the trait free of the async-trait machinery. [`EvolutionRailAdapter`] is the
//! async bridge into [`crate::rails::HarnessRail`]; it logs and swallows hook
//! failures, so an evolution problem can never abort a run.
use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::Result;

use super::rail_priority;
use super::{HarnessRail, HarnessRailContext};

/// A self-evolution hook fired by the harness at coarse lifecycle points.
///
/// Every hook has a no-op default, so an implementation overrides only what it
/// needs. A returned `Err` is logged by [`EvolutionRailAdapter`] and does not
/// abort the run.
pub trait EvolutionRail: Send + Sync {
    /// Stable name of the rail.
    fn name(&self) -> &str;

    /// Dispatch priority; defaults to [`rail_priority::EVOLUTION`].
    fn priority(&self) -> i32 {
        rail_priority::EVOLUTION
    }

    /// A new invocation starts.
    ///
    /// # Errors
    ///
    /// Returns an error the adapter logs and swallows.
    fn on_before_invoke(&self, _ctx: &mut HarnessRailContext) -> Result<()> {
        Ok(())
    }

    /// One outer task-loop iteration finished.
    ///
    /// # Errors
    ///
    /// Returns an error the adapter logs and swallows.
    fn on_after_task_iteration(&self, _ctx: &mut HarnessRailContext) -> Result<()> {
        Ok(())
    }
}

/// An [`EvolutionRail`] that does nothing; the default registration.
#[derive(Debug, Clone, Default)]
pub struct NoopEvolutionRail;

impl EvolutionRail for NoopEvolutionRail {
    fn name(&self) -> &'static str {
        "evolution"
    }
}

/// Presents an [`EvolutionRail`] as a [`HarnessRail`], isolating its failures.
pub struct EvolutionRailAdapter {
    inner: Arc<dyn EvolutionRail>,
}

impl EvolutionRailAdapter {
    /// Wraps `inner`.
    #[must_use]
    pub fn new(inner: Arc<dyn EvolutionRail>) -> Self {
        Self { inner }
    }

    /// The wrapped rail.
    #[must_use]
    pub fn inner(&self) -> &Arc<dyn EvolutionRail> {
        &self.inner
    }
}

#[allow(clippy::missing_fields_in_debug)]
impl fmt::Debug for EvolutionRailAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EvolutionRailAdapter")
            .field("rail", &self.inner.name())
            .finish()
    }
}

#[async_trait::async_trait]
impl HarnessRail for EvolutionRailAdapter {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn priority(&self) -> i32 {
        self.inner.priority()
    }

    async fn before_invoke(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        if let Err(error) = self.inner.on_before_invoke(ctx) {
            tracing::warn!(
                rail = self.inner.name(),
                hook = "before_invoke",
                error = %error,
                "evolution rail hook failed"
            );
        }
        Ok(())
    }

    async fn after_task_iteration(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        if let Err(error) = self.inner.on_after_task_iteration(ctx) {
            tracing::warn!(
                rail = self.inner.name(),
                hook = "after_task_iteration",
                error = %error,
                "evolution rail hook failed"
            );
        }
        Ok(())
    }
}

/// Wraps an [`EvolutionRail`] into a registerable [`HarnessRail`].
#[must_use]
pub fn evolution_adapter(rail: Arc<dyn EvolutionRail>) -> Arc<dyn HarnessRail> {
    Arc::new(EvolutionRailAdapter::new(rail))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use openjiuwen_agent::RailEvent;
    use std::sync::Mutex;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Call {
        BeforeInvoke,
        AfterTaskIteration,
    }

    /// Records every hook it observes; optionally fails one of them.
    struct RecordingEvolution {
        calls: Mutex<Vec<Call>>,
        fail: bool,
    }

    impl RecordingEvolution {
        fn new(fail: bool) -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                fail,
            }
        }

        fn calls(&self) -> Vec<Call> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl EvolutionRail for RecordingEvolution {
        fn name(&self) -> &'static str {
            "recording-evolution"
        }

        fn on_before_invoke(&self, _ctx: &mut HarnessRailContext) -> Result<()> {
            self.calls.lock().unwrap().push(Call::BeforeInvoke);
            if self.fail {
                return Err(openjiuwen_common::error::Error::unsupported("nope"));
            }
            Ok(())
        }

        fn on_after_task_iteration(&self, _ctx: &mut HarnessRailContext) -> Result<()> {
            self.calls.lock().unwrap().push(Call::AfterTaskIteration);
            if self.fail {
                return Err(openjiuwen_common::error::Error::unsupported("nope"));
            }
            Ok(())
        }
    }

    #[test]
    fn noop_rail_has_the_evolution_name_and_priority() {
        let rail = NoopEvolutionRail;
        assert_eq!(rail.name(), "evolution");
        assert_eq!(rail.priority(), rail_priority::EVOLUTION);
        assert_eq!(rail.priority(), 60);

        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        rail.on_before_invoke(&mut ctx).unwrap();
        rail.on_after_task_iteration(&mut ctx).unwrap();
    }

    #[tokio::test]
    async fn adapter_forwards_both_hooks() {
        let rail = Arc::new(RecordingEvolution::new(false));
        let adapter = EvolutionRailAdapter::new(rail.clone());

        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        adapter.before_invoke(&mut ctx).await.unwrap();
        adapter.after_task_iteration(&mut ctx).await.unwrap();

        assert_eq!(rail.calls(), [Call::BeforeInvoke, Call::AfterTaskIteration]);
        assert_eq!(adapter.name(), "recording-evolution");
        assert_eq!(adapter.priority(), rail_priority::EVOLUTION);
        assert_eq!(adapter.inner().name(), "recording-evolution");
    }

    #[tokio::test]
    async fn adapter_swallows_hook_failures() {
        let rail = Arc::new(RecordingEvolution::new(true));
        let adapter = EvolutionRailAdapter::new(rail.clone());

        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        adapter.before_invoke(&mut ctx).await.unwrap();
        adapter.after_task_iteration(&mut ctx).await.unwrap();

        assert_eq!(rail.calls().len(), 2);
    }

    #[tokio::test]
    async fn adapter_leaves_the_other_harness_hooks_as_no_ops() {
        let rail = Arc::new(RecordingEvolution::new(false));
        let adapter = EvolutionRailAdapter::new(rail.clone());

        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        adapter.before_model_call(&mut ctx).await.unwrap();
        adapter.before_task_iteration(&mut ctx).await.unwrap();

        assert_eq!(rail.calls().len(), 0);
    }

    #[test]
    fn evolution_adapter_returns_a_harness_rail() {
        let rail: Arc<dyn HarnessRail> = evolution_adapter(Arc::new(NoopEvolutionRail));
        assert_eq!(rail.name(), "evolution");
        assert_eq!(rail.priority(), 60);
    }

    #[test]
    fn debug_prints_the_rail_name() {
        let adapter = EvolutionRailAdapter::new(Arc::new(NoopEvolutionRail));
        assert_eq!(
            format!("{adapter:?}"),
            "EvolutionRailAdapter { rail: \"evolution\" }"
        );
    }
}
