//! The adopt family: decide whether a tuned prompt is worth installing.
//!
//! Ports `openjiuwen/dev_tools/tune/optimizer/adopt/` (`adopt_optimizer.py`,
//! `utils.py`) as a deterministic decision layer over the existing prompt-search
//! seams: a policy turns a reward history into an adoption decision, the memory
//! records what was adopted, and the service composes drift, convergence, and
//! memory so a caller can adopt a candidate without re-implementing them.
//!
//! Python's `AdoptOptimizer` rewrites every `LLMCall` of a multi-node workflow in
//! place — conclude each node's job, critique the whole-workflow output, derive a
//! corrected node output, then compute a per-node "textual gradient" through a
//! chain of model prompts — and `PartialOptimizer` is the instruction optimizer
//! it uses for one parameter. None of that survives as-is: there is no
//! `LLMCall`/`TextualParameter` to mutate, no `OptimizeHistory` to read traced
//! node pairs from, and no prompt bank to render. What survives is the part that
//! makes the optimizer useful and testable — *should this candidate replace the
//! installed prompt, and on what evidence?* — which the port states once, purely,
//! and reuses.
//!
//! - [`AdoptConfig`] / [`AdoptPolicy`] / [`AdoptDecision`] / [`AdoptReason`] —
//!   the thresholds and the pure verdict (correctness gate → drift gate →
//!   history → improvement).
//! - [`AdoptMemory`] — this run's adopted/rejected decisions plus the optional
//!   [`PromptMemory`] they are archived into.
//! - [`AdoptService`] / [`AdoptOutcome`] — measure (drift judge, convergence,
//!   correctness), decide, and archive in one `async fn`.
//! - [`select_bad_cases`] / [`select_good_cases`] /
//!   [`describe_gradient_context`] — Python's good/bad case split and the
//!   textual-gradient artifact, made deterministic.
//! - [`GradientPolicy`] / [`DeterministicGradientPolicy`] — the seam that lets a
//!   caller opt into a model-backed gradient later.
//!
//! Everything is deterministic: no clock (`now` is always a parameter), no
//! randomness (Python's `random.sample` becomes a case-id ordering), no
//! environment reads, no sleeps, and no network. The only `async fn` is
//! [`AdoptService::decide`] and the [`GradientPolicy`] trait, because both may
//! call an injected seam that a caller implements over a model.
//!
//! [`PromptMemory`]: crate::tune::prompt_search::PromptMemory
//!
//! # Example
//!
//! The snippet is a usage sketch rather than a compiled doctest: it reaches the
//! family through `openjiuwen_dev_tools::tune`, which re-exports it only once the
//! module wiring outside this node lists it.
//!
//! ```text
//! use openjiuwen_dev_tools::tune::{
//!     AdoptPolicy, AdoptReason, AdoptService, PromptCandidate, PromptTaskSpec,
//! };
//!
//! let policy = AdoptPolicy::new();
//! let candidate = PromptCandidate::new("Answer {{question}} briefly.", "seed");
//!
//! // No history: the first candidate becomes the baseline.
//! let first = policy.decide(0.4, 1.0, 0.0, &[]);
//! assert!(first.adopted());
//! assert_eq!(first.reason(), &AdoptReason::FirstCandidate);
//!
//! // A later candidate only wins on a measurable gain.
//! assert!(policy.decide(0.8, 1.0, 0.0, &[0.4]).adopted());
//! assert!(!policy.decide(0.41, 1.0, 0.0, &[0.4]).adopted());
//!
//! // The async service measures, gates, decides, and archives in one call.
//! async fn sketch() -> Result<(), openjiuwen_common::error::Error> {
//! let service = AdoptService::new(policy);
//! let outcome = service
//!     .decide(
//!         &PromptTaskSpec::new("answer arithmetic questions"),
//!         &candidate,
//!         0.6,
//!         &[1.0, 0.5],
//!         &[],
//!         1_700_000_000.0,
//!     )
//!     .await?;
//! assert!(outcome.adopted());
//! Ok(())
//! }
//! ```

// The family is re-exported by the optimizer facade (`pub use adopt::*;` in
// `../optimizer.rs`), but `crate::tune` still lists its re-exports explicitly, so
// nothing here is reachable from outside the crate yet. Keeping the real `pub`
// visibility (rather than `pub(crate)`) is deliberate: the surface is the
// deliverable. Both lints fall silent as soon as `crate::tune` re-exports the
// family; this allow exists only for that window.
#![allow(dead_code, unreachable_pub)]

mod cases;
mod gradient;
mod memory;
mod policy;
mod service;

#[cfg(test)]
mod support;
#[cfg(test)]
mod tests;

// A `pub use` of an item nothing can reach yet counts as unused; the same missing
// wiring as the module-level allow above, so it is silenced per statement rather
// than for the whole module (a genuinely unused import must still warn).
#[allow(unused_imports)]
pub use cases::{
    describe_gradient_context, select_bad_cases, select_good_cases, DEFAULT_BAD_CASES_SAMPLE_NUM,
};
#[allow(unused_imports)]
pub use gradient::{render_gradient, DeterministicGradientPolicy, GradientPolicy};
#[allow(unused_imports)]
pub use memory::AdoptMemory;
#[allow(unused_imports)]
pub use policy::{
    AdoptConfig, AdoptDecision, AdoptPolicy, AdoptReason, DEFAULT_ADOPT_MAX_DRIFT,
    DEFAULT_ADOPT_MIN_CORRECTNESS, DEFAULT_ADOPT_MIN_GAIN, DEFAULT_ADOPT_MIN_HISTORY,
};
#[allow(unused_imports)]
pub use service::{AdoptOutcome, AdoptService};
