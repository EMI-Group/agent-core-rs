//! The self-evolution rail seam and its harness-side bodies.
//!
//! The evolution bodies that need a research or an orchestrator stack (RSI,
//! `agent_evolving`, skill-evolution proposals) live in crates *above* this one,
//! so the harness depends on this small trait instead of on any evolution
//! crate: an upper layer implements [`EvolutionRail`], wraps it with
//! [`evolution_adapter`], and registers the result like any other rail.
//!
//! Two kinds of rail live here:
//!
//! - [`EvolutionRail`] is the seam. [`NoopEvolutionRail`] is the default
//!   registration; upper layers bring the real behaviour.
//! - [`MetricsEvolutionRail`] is the harness-side body that needs no upper
//!   layer: it records one [`EvolutionMetrics`] row per outer task-loop
//!   iteration, keeps a bounded [`ProposalLog`], and writes an
//!   [`EvolutionSnapshot`] when a path is configured.
//!
//! The hooks are synchronous by design — evolution work is metadata bookkeeping
//! (proposal drafting, metric sampling) with no model or tool IO — which keeps
//! the trait free of the async-trait machinery. [`EvolutionRailAdapter`] is the
//! async bridge into [`crate::rails::HarnessRail`]; it logs and swallows hook
//! failures, so an evolution problem can never abort a run.
mod metrics;
mod proposal;
mod rail;
mod snapshot;
mod store;

use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::Result;

use super::rail_priority;
use super::{HarnessRail, HarnessRailContext};

pub use metrics::{default_metrics_clock, EvolutionMetrics, IterationRecorder, MetricsClock};
pub use proposal::{EvolutionProposal, ProposalLog};
pub use rail::{MetricsEvolutionRail, MetricsEvolutionRailConfig};
pub use snapshot::{EvolutionSnapshot, EVOLUTION_SNAPSHOT_VERSION};

/// A self-evolution hook fired by the harness at coarse lifecycle points.
///
/// Every hook has a no-op default, so an implementation overrides only what it
/// needs. A returned `Err` is logged by [`EvolutionRailAdapter`] and does not
/// abort the run. Implementations are shared across invocations, so per-run
/// state belongs in the context or behind interior mutability, never in a
/// freshly assumed `&mut self`.
///
/// The hooks are synchronous and must not block: read the context, update
/// in-memory counters, and leave file or network IO to the owner of the rail.
pub trait EvolutionRail: Send + Sync {
    /// Stable name of the rail.
    fn name(&self) -> &str;

    /// Dispatch priority; defaults to [`rail_priority::EVOLUTION`].
    fn priority(&self) -> i32 {
        rail_priority::EVOLUTION
    }

    /// A new invocation starts, before any message is built.
    ///
    /// The context holds the run identity (`base.agent_id`, `base.session`) and
    /// an empty `base.extra`; this is where a rail mints its run id and resets
    /// per-run counters.
    ///
    /// # Errors
    ///
    /// Returns an error the adapter logs and swallows.
    fn on_before_invoke(&self, _ctx: &mut HarnessRailContext) -> Result<()> {
        Ok(())
    }

    /// The model returned a reply (`base.event == AfterModelCall`).
    ///
    /// `base.response` is set: its `message` may carry tool calls, `usage`
    /// holds the provider token accounting, and `finish_reason` the provider
    /// stop reason. `base.iteration` is the current loop iteration and
    /// `base.extra` still holds everything earlier rails published.
    ///
    /// # Errors
    ///
    /// Returns an error the adapter logs and swallows.
    fn on_after_model_call(&self, _ctx: &mut HarnessRailContext) -> Result<()> {
        Ok(())
    }

    /// One tool call returned (`base.event == AfterToolCall`).
    ///
    /// `base.tool_call` holds the call being executed (its `name`,
    /// `arguments`, `id`) and `base.tool_result` the outcome (`success`,
    /// `error`, `output`, `text`); a loop may fire the hook with only one of
    /// the two populated, so read them defensively.
    ///
    /// # Errors
    ///
    /// Returns an error the adapter logs and swallows.
    fn on_after_tool_call(&self, _ctx: &mut HarnessRailContext) -> Result<()> {
        Ok(())
    }

    /// One outer task-loop iteration finished.
    ///
    /// `ctx.task` is set (`TaskIterationInfo { round_id, iteration,
    /// is_follow_up, loop_event }`), the kernel messages already contain the
    /// tool observations of the iteration, and `base.iteration` is the inner
    /// `ReAct` iteration. A run that never entered the task loop never fires
    /// this hook. `base.event` is not a reliable phase discriminator here: the
    /// loop reuses a base context, so read `ctx.task` instead.
    ///
    /// # Errors
    ///
    /// Returns an error the adapter logs and swallows.
    fn on_after_task_iteration(&self, _ctx: &mut HarnessRailContext) -> Result<()> {
        Ok(())
    }

    /// The invocation finished (`base.event == AfterInvoke`).
    ///
    /// `base.output` is the terminal [`openjiuwen_agent::AgentOutput`] (answer,
    /// error or interrupt, plus its `termination_reason`/`finish_reason`); the
    /// session, agent id and `extra` of the whole run are still available. This
    /// is the last evolution firing point of a run.
    ///
    /// # Errors
    ///
    /// Returns an error the adapter logs and swallows.
    fn on_after_invoke(&self, _ctx: &mut HarnessRailContext) -> Result<()> {
        Ok(())
    }
}

/// An [`EvolutionRail`] that does nothing; the default registration.
///
/// It inherits every no-op default and the [`rail_priority::EVOLUTION`]
/// priority, so registering it is the same as registering no evolution rail at
/// all — except that the name is taken.
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

    /// Runs one synchronous hook, logging and swallowing its failure.
    fn isolate(
        &self,
        hook: &'static str,
        call: impl FnOnce(&mut HarnessRailContext) -> Result<()>,
        ctx: &mut HarnessRailContext,
    ) {
        if let Err(error) = call(ctx) {
            tracing::warn!(
                rail = self.inner.name(),
                hook,
                error = %error,
                "evolution rail hook failed"
            );
        }
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
        self.isolate("before_invoke", |ctx| self.inner.on_before_invoke(ctx), ctx);
        Ok(())
    }

    async fn after_model_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        self.isolate(
            "after_model_call",
            |ctx| self.inner.on_after_model_call(ctx),
            ctx,
        );
        Ok(())
    }

    async fn after_tool_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        self.isolate(
            "after_tool_call",
            |ctx| self.inner.on_after_tool_call(ctx),
            ctx,
        );
        Ok(())
    }

    async fn after_task_iteration(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        self.isolate(
            "after_task_iteration",
            |ctx| self.inner.on_after_task_iteration(ctx),
            ctx,
        );
        Ok(())
    }

    async fn after_invoke(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        self.isolate("after_invoke", |ctx| self.inner.on_after_invoke(ctx), ctx);
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
    enum Hook {
        BeforeInvoke,
        AfterModelCall,
        AfterToolCall,
        AfterTaskIteration,
        AfterInvoke,
    }

    /// Records every hook it observes; optionally fails all of them.
    struct RecordingEvolution {
        calls: Mutex<Vec<Hook>>,
        fail: bool,
    }

    impl RecordingEvolution {
        fn new(fail: bool) -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                fail,
            }
        }

        fn calls(&self) -> Vec<Hook> {
            self.calls.lock().unwrap().clone()
        }

        fn record(&self, call: Hook) -> Result<()> {
            self.calls.lock().unwrap().push(call);
            if self.fail {
                return Err(openjiuwen_common::error::Error::unsupported("nope"));
            }
            Ok(())
        }
    }

    impl EvolutionRail for RecordingEvolution {
        fn name(&self) -> &'static str {
            "recording-evolution"
        }

        fn on_before_invoke(&self, _ctx: &mut HarnessRailContext) -> Result<()> {
            self.record(Hook::BeforeInvoke)
        }

        fn on_after_model_call(&self, _ctx: &mut HarnessRailContext) -> Result<()> {
            self.record(Hook::AfterModelCall)
        }

        fn on_after_tool_call(&self, _ctx: &mut HarnessRailContext) -> Result<()> {
            self.record(Hook::AfterToolCall)
        }

        fn on_after_task_iteration(&self, _ctx: &mut HarnessRailContext) -> Result<()> {
            self.record(Hook::AfterTaskIteration)
        }

        fn on_after_invoke(&self, _ctx: &mut HarnessRailContext) -> Result<()> {
            self.record(Hook::AfterInvoke)
        }
    }

    #[test]
    fn noop_rail_is_a_faithful_no_op_on_every_hook() {
        let rail = NoopEvolutionRail;
        assert_eq!(rail.name(), "evolution");
        assert_eq!(rail.priority(), rail_priority::EVOLUTION);
        assert_eq!(rail.priority(), 60);

        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        rail.on_before_invoke(&mut ctx).unwrap();
        rail.on_after_model_call(&mut ctx).unwrap();
        rail.on_after_tool_call(&mut ctx).unwrap();
        rail.on_after_task_iteration(&mut ctx).unwrap();
        rail.on_after_invoke(&mut ctx).unwrap();

        assert!(
            ctx.extra().is_empty(),
            "the no-op rail touches no context channel"
        );
        assert!(ctx.base.response.is_none());
        assert!(ctx.base.output.is_none());
        assert!(ctx.base.tool_result.is_none());
        assert!(ctx.base.force_finish.is_none());
        assert!(ctx.base.skip_tool.is_none());
        assert!(!ctx.base.model_continue);
        assert!(!ctx.is_interrupted());
    }

    #[tokio::test]
    async fn adapter_forwards_every_hook() {
        let rail = Arc::new(RecordingEvolution::new(false));
        let adapter = EvolutionRailAdapter::new(rail.clone());

        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        adapter.before_invoke(&mut ctx).await.unwrap();
        adapter.after_model_call(&mut ctx).await.unwrap();
        adapter.after_tool_call(&mut ctx).await.unwrap();
        adapter.after_task_iteration(&mut ctx).await.unwrap();
        adapter.after_invoke(&mut ctx).await.unwrap();

        assert_eq!(
            rail.calls(),
            [
                Hook::BeforeInvoke,
                Hook::AfterModelCall,
                Hook::AfterToolCall,
                Hook::AfterTaskIteration,
                Hook::AfterInvoke,
            ]
        );
        assert_eq!(adapter.name(), "recording-evolution");
        assert_eq!(adapter.priority(), rail_priority::EVOLUTION);
        assert_eq!(adapter.inner().name(), "recording-evolution");
    }

    #[tokio::test]
    async fn adapter_swallows_hook_failures_on_every_hook() {
        let rail = Arc::new(RecordingEvolution::new(true));
        let adapter = EvolutionRailAdapter::new(rail.clone());

        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        adapter.before_invoke(&mut ctx).await.unwrap();
        adapter.after_model_call(&mut ctx).await.unwrap();
        adapter.after_tool_call(&mut ctx).await.unwrap();
        adapter.after_task_iteration(&mut ctx).await.unwrap();
        adapter.after_invoke(&mut ctx).await.unwrap();

        assert_eq!(rail.calls().len(), 5, "every hook ran despite its errors");
    }

    #[tokio::test]
    async fn adapter_leaves_the_other_harness_hooks_as_no_ops() {
        let rail = Arc::new(RecordingEvolution::new(false));
        let adapter = EvolutionRailAdapter::new(rail.clone());

        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        adapter.before_model_call(&mut ctx).await.unwrap();
        adapter.before_tool_call(&mut ctx).await.unwrap();
        adapter.on_model_exception(&mut ctx).await.unwrap();
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
