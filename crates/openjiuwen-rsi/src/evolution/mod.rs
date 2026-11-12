//! Shared candidate, mutation, evaluation, and selection protocol.
//!
//! Both RSI halves — the harness self-optimisation loop
//! ([`crate::harness_rsi::auto_harness`]) and the program search
//! ([`crate::artifact_rsi::program_opt`]) — rank candidates the same way: a
//! [`Candidate`] carries a text body, a [`MutationStrategy`] derives children
//! from parents, an [`Evaluator`] scores each child against a set of [`Case`]s,
//! and a [`SelectionStrategy`] keeps the elite.
//!
//! The protocol is deliberately free of ML, harness, and I/O dependencies so a
//! fake evaluator or a different search strategy plugs in without touching the
//! orchestrator. Every public type is `serde`-serializable (`snake_case`,
//! default-tolerant) and deterministic: ids, ordering, and tie-breaks depend
//! only on the inputs.
//!
//! [`MutationStrategy`]: mutation::MutationStrategy
//! [`Evaluator`]: evaluation::Evaluator
//! [`Case`]: evaluation::Case

pub mod candidate;
pub mod evaluation;
pub mod mutation;
pub mod selection;

pub use candidate::{Candidate, CandidateId, CandidateStatus};
pub use evaluation::{Case, CaseResult, EvaluationReport, Evaluator};
pub use mutation::{DeterministicMutator, MutationRequest, MutationStrategy};
pub use selection::{rank_score, ranked, ElitistSelection, SelectionStrategy};
