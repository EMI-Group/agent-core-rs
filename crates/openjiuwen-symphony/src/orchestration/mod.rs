//! Deterministic relation candidates, planning, and plan execution.
//!
//! The orchestration layer answers the engine's central question — given a user
//! request, which capabilities must run, and in what order — entirely offline and
//! deterministically:
//!
//! - [`candidates`] derives `can_feed` relation candidates between capabilities with
//!   three deterministic rules (exact IO match, semantic text overlap, textual
//!   coercion), so the graph can be built without an LLM matcher.
//! - [`planner`] runs a bounded graph search over the capability graph, scoring plans
//!   with the Python formula and a pluggable [`planner::SemanticScorer`]. The default
//!   scorer is lexical, so planning is reproducible.
//! - [`plan`] is the typed, serializable result ([`plan::OrchestrationPlan`]) with its
//!   JGF projection and Kahn [`plan::OrchestrationPlan::stages`].
//! - [`executor`] compiles a plan into an [`openjiuwen_graph::CompiledGraph`] and runs
//!   it through a [`executor::CapabilityRunner`], recording per-step outcomes.
//!
//! Every ordering in this module is explicit: candidates sort by strategy rank,
//! confidence, source, and target; plans rank on a fixed tuple; graph node ids are
//! qualified capability ids; ties break on identifiers, never on hash-map order.

pub mod candidates;
pub mod executor;
pub mod plan;
pub mod planner;
