//! Deterministic relation candidates, planning, and plan execution.
//!
//! The orchestration layer answers the engine's central question — given a user
//! request, which capabilities must run, and in what order — entirely offline and
//! deterministically:
//!
//! - [`candidates`](crate::orchestration::candidates) derives `can_feed` relation
//!   candidates between capabilities with three deterministic rules (exact IO match,
//!   semantic text overlap, textual coercion), so the graph can be built without an LLM
//!   matcher.
//! - [`planner`](crate::orchestration::planner) runs a bounded graph search over the
//!   capability graph, scoring plans with the Python formula and a pluggable
//!   [`planner::SemanticScorer`](crate::orchestration::planner::SemanticScorer). The
//!   default scorer is lexical, so planning is reproducible.
//! - [`plan`](crate::orchestration::plan) is the typed, serializable result
//!   ([`plan::OrchestrationPlan`](crate::orchestration::plan::OrchestrationPlan)) with its
//!   JGF projection and Kahn
//!   [`plan::OrchestrationPlan::stages`](crate::orchestration::plan::OrchestrationPlan::stages).
//! - [`executor`](crate::orchestration::executor) compiles a plan into an
//!   [`openjiuwen_graph::CompiledGraph`] and runs it through a
//!   [`executor::CapabilityRunner`](crate::orchestration::executor::CapabilityRunner),
//!   recording per-step outcomes.
//!
//! Every ordering in this module is explicit: candidates sort by strategy rank,
//! confidence, source, and target; plans rank on a fixed tuple; graph node ids are
//! qualified capability ids; ties break on identifiers, never on hash-map order.

pub mod candidates;
pub mod executor;
pub mod plan;
pub mod planner;

pub use candidates::{CandidateConfig, CandidateGenerator, MatchStrategy, RelationCandidate};
pub use executor::{
    CapabilityRunner, ExecutorConfig, PlanExecution, PlanExecutor, RunnerContext, StepOutcome,
    UnsupportedRunner,
};
pub use plan::{OrchestrationPlan, PlanEdge, PlanStatus, PlanStep};
pub use planner::{
    normalize_language, plan_title, LexicalSemanticScorer, PlanConfig, PlanMode, PlanRequest,
    Planner, SemanticScorer,
};
