//! Pure evolution events and the harness rail that publishes them.
//!
//! The dependency arrow points `openjiuwen-agent-evolving -> openjiuwen-harness`,
//! never the other way: the harness only knows the
//! [`openjiuwen_harness::EvolutionRail`] seam and its
//! [`evolution_adapter`], and this crate is the upper layer that implements it.
//! Keeping the translation here means the harness stays free of evolution types,
//! while a harness run can still report what happened to any
//! [`EvolutionObserver`] — a metrics sink, an experience store, or a test
//! recorder.
//!
//! [`EvolutionEvent`] is deliberately pure data: no locks, no IO, no clocks. The
//! [`HarnessEvolutionRail`] only extracts ids and iteration counters from the
//! harness context and forwards them, so an observer decides what to do with them.
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use openjiuwen_common::error::Result;
use openjiuwen_harness::rails::evolution_adapter;
use openjiuwen_harness::{EvolutionRail, HarnessRail, HarnessRailContext};

/// Dispatch priority of the evolution rail.
///
/// Mirrors [`openjiuwen_harness::rail_priority::EVOLUTION`], so the rail sorts
/// into the same slot whether it is registered through the harness constant or
/// through this crate.
pub const EVOLUTION_PRIORITY: i32 = 60;

/// A coarse self-evolution event emitted by the harness rail.
///
/// The event is serialized with a `type` tag (`invoke_started`,
/// `task_iteration_completed`), matching the Python event payloads that downstream
/// collectors consume.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EvolutionEvent {
    /// A new agent invocation started.
    InvokeStarted {
        /// Id of the agent being invoked.
        agent_id: String,
        /// Session the invocation runs in, when the caller supplied one.
        session_id: Option<String>,
        /// Current loop iteration, `1`-based.
        iteration: usize,
    },
    /// One outer task-loop iteration finished.
    TaskIterationCompleted {
        /// Id of the agent whose task loop finished the iteration.
        agent_id: String,
        /// Session the iteration ran in, when the caller supplied one.
        session_id: Option<String>,
        /// Id of the outer task round.
        round_id: u64,
        /// Iteration index inside the round, `0`-based.
        iteration: usize,
        /// Whether the iteration ran a follow-up message rather than the task.
        is_follow_up: bool,
    },
}

/// A sink for [`EvolutionEvent`]s.
///
/// Observers are synchronous by design: the harness evolution hooks are metadata
/// bookkeeping, and the seam must stay free of the async-trait machinery.
pub trait EvolutionObserver: Send + Sync {
    /// Records one event.
    ///
    /// # Errors
    ///
    /// Returns an error when the observer cannot record the event; the rail
    /// forwards it unchanged and the harness adapter logs and swallows it.
    fn observe(&self, event: &EvolutionEvent) -> Result<()>;
}

/// Static description of the evolution rail, mirroring Python's rail config.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EvolutionRailSpec {
    /// Rail name used for registration and diagnostics.
    pub name: String,
    /// Dispatch priority; see [`EVOLUTION_PRIORITY`].
    pub priority: i32,
    /// Whether the rail should be registered.
    pub enabled: bool,
}

impl Default for EvolutionRailSpec {
    fn default() -> Self {
        Self {
            name: "evolution".to_owned(),
            priority: EVOLUTION_PRIORITY,
            enabled: true,
        }
    }
}

impl EvolutionRailSpec {
    /// The rail name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The dispatch priority.
    #[must_use]
    pub fn priority(&self) -> i32 {
        self.priority
    }

    /// Whether the rail is enabled.
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Replaces the rail name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Replaces the dispatch priority.
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Replaces the enabled flag.
    #[must_use]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

/// The harness evolution rail: translates harness hooks into [`EvolutionEvent`]s.
///
/// Implements [`openjiuwen_harness::EvolutionRail`]; wrap it with
/// [`registerable_rail`] to mount it in a [`openjiuwen_harness::HarnessRailRegistry`].
pub struct HarnessEvolutionRail {
    spec: EvolutionRailSpec,
    observer: Arc<dyn EvolutionObserver>,
}

impl HarnessEvolutionRail {
    /// Creates a rail with the default [`EvolutionRailSpec`] feeding `observer`.
    #[must_use]
    pub fn new(observer: Arc<dyn EvolutionObserver>) -> Self {
        Self {
            spec: EvolutionRailSpec::default(),
            observer,
        }
    }

    /// Replaces the rail spec.
    #[must_use]
    pub fn with_spec(mut self, spec: EvolutionRailSpec) -> Self {
        self.spec = spec;
        self
    }

    /// The configured spec.
    #[must_use]
    pub fn spec(&self) -> &EvolutionRailSpec {
        &self.spec
    }

    /// The observer the rail forwards events to.
    #[must_use]
    pub fn observer(&self) -> &Arc<dyn EvolutionObserver> {
        &self.observer
    }

    /// Builds the event of a task-loop hook from `ctx`.
    fn task_iteration_event(ctx: &HarnessRailContext) -> EvolutionEvent {
        let (round_id, iteration, is_follow_up) = ctx
            .task_info()
            .map_or((0_u64, ctx.base.iteration, false), |info| {
                (info.round_id, info.iteration, info.is_follow_up)
            });
        EvolutionEvent::TaskIterationCompleted {
            agent_id: ctx.base.agent_id.clone(),
            session_id: ctx.base.session.as_ref().map(|s| s.id().to_owned()),
            round_id,
            iteration,
            is_follow_up,
        }
    }
}

impl EvolutionRail for HarnessEvolutionRail {
    fn name(&self) -> &str {
        &self.spec.name
    }

    fn priority(&self) -> i32 {
        self.spec.priority
    }

    fn on_before_invoke(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        let event = EvolutionEvent::InvokeStarted {
            agent_id: ctx.base.agent_id.clone(),
            session_id: ctx.base.session.as_ref().map(|s| s.id().to_owned()),
            iteration: ctx.base.iteration,
        };
        self.observer.observe(&event)
    }

    fn on_after_task_iteration(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        let event = Self::task_iteration_event(ctx);
        self.observer.observe(&event)
    }
}

/// Wraps a [`HarnessEvolutionRail`] into a registerable
/// [`openjiuwen_harness::HarnessRail`].
///
/// The adapter logs and swallows hook errors, so an evolution failure can never
/// abort a harness run.
#[must_use]
pub fn registerable_rail(rail: HarnessEvolutionRail) -> Arc<dyn HarnessRail> {
    evolution_adapter(Arc::new(rail))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use openjiuwen_core::single_agent::RailEvent;
    use openjiuwen_harness::rail_priority;

    struct Recorder {
        events: Mutex<Vec<EvolutionEvent>>,
    }

    impl EvolutionObserver for Recorder {
        fn observe(&self, event: &EvolutionEvent) -> Result<()> {
            self.events.lock().expect("lock").push(event.clone());
            Ok(())
        }
    }

    fn recorder() -> Arc<Recorder> {
        Arc::new(Recorder {
            events: Mutex::new(Vec::new()),
        })
    }

    #[test]
    fn priority_matches_the_harness_ladder() {
        assert_eq!(EVOLUTION_PRIORITY, 60);
        assert_eq!(EVOLUTION_PRIORITY, rail_priority::EVOLUTION);
    }

    #[test]
    fn default_spec_is_named_evolution_and_enabled() {
        let spec = EvolutionRailSpec::default();
        assert_eq!(spec.name(), "evolution");
        assert_eq!(spec.priority(), EVOLUTION_PRIORITY);
        assert!(spec.enabled());
        assert_eq!(spec, HarnessEvolutionRail::new(recorder()).spec().clone());
    }

    #[test]
    fn builders_replace_the_spec_fields() {
        let spec = EvolutionRailSpec::default()
            .with_name("custom")
            .with_priority(7)
            .with_enabled(false);
        assert_eq!(spec.name(), "custom");
        assert_eq!(spec.priority(), 7);
        assert!(!spec.enabled());
    }

    #[test]
    fn before_invoke_records_the_extracted_fields() {
        let sink = recorder();
        let rail = HarnessEvolutionRail::new(sink.clone());
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        ctx.base.agent_id = "agent-1".to_owned();
        ctx.base.iteration = 2;

        rail.on_before_invoke(&mut ctx).expect("observe");

        let events = sink.events.lock().expect("lock");
        assert_eq!(
            events.as_slice(),
            [EvolutionEvent::InvokeStarted {
                agent_id: "agent-1".to_owned(),
                session_id: None,
                iteration: 2,
            }]
        );
    }

    #[test]
    fn event_serializes_with_a_type_tag() {
        let event = EvolutionEvent::InvokeStarted {
            agent_id: "agent-1".to_owned(),
            session_id: Some("sess-1".to_owned()),
            iteration: 2,
        };
        let json = serde_json::to_value(&event).expect("serialize");
        assert_eq!(
            json,
            serde_json::json!({
                "type": "invoke_started",
                "agent_id": "agent-1",
                "session_id": "sess-1",
                "iteration": 2,
            })
        );

        let decoded: EvolutionEvent = serde_json::from_value(json).expect("deserialize");
        assert_eq!(decoded, event);
    }

    #[test]
    fn task_iteration_without_info_falls_back_to_the_base_iteration() {
        let sink = recorder();
        let rail = HarnessEvolutionRail::new(sink.clone());
        let mut ctx = HarnessRailContext::for_event(RailEvent::AfterReactIteration);
        ctx.base.agent_id = "agent-1".to_owned();
        ctx.base.iteration = 4;

        rail.on_after_task_iteration(&mut ctx).expect("observe");

        let events = sink.events.lock().expect("lock");
        assert_eq!(
            events.as_slice(),
            [EvolutionEvent::TaskIterationCompleted {
                agent_id: "agent-1".to_owned(),
                session_id: None,
                round_id: 0,
                iteration: 4,
                is_follow_up: false,
            }]
        );
    }
}
