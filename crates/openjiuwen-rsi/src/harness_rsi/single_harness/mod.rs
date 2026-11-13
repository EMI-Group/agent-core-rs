//! Single-harness optimisation: iterative and continuous promotion loops over
//! an injected evaluator and improver.
//!
//! Maps to Python `openjiuwen.rsi.harness_rsi.single_harness`. The Python
//! package is a 200k-line harness driver: it materialises source evidence,
//! runs `DeepAgent` turns, shells out to pytest, and writes evaluation
//! references. The Rust port owns the deterministic orchestration core of the
//! two loops and leaves the model-driven halves behind the crate's seams:
//!
//! | Python | Rust |
//! |---|---|
//! | `iterative.run_single_harness` | [`IterativeSingleHarness`] (epoch checkpoint per batch) |
//! | `continuous.run_continuous` | [`ContinuousSingleHarness`] (two-stage, attempt-based) |
//! | `promotion_policy.py` | [`promotion`] (pure promotion math) |
//! | evaluator + case runner | [`crate::evolution::Evaluator`] |
//! | candidate feedback / member optimiser (LLM) | [`crate::evolution::MutationStrategy`] |
//! | `state.yaml` epoch checkpoint | [`state::SingleHarnessState`] |
//!
//! The analyser and improver LLM turns are *not* implemented here: the crate's
//! deferred-scope table records `harness_rsi.{evaluation_result_analyzer,
//! member_optimizer}` as agent-driven surfaces, so this module asks a
//! [`MutationStrategy`](crate::evolution::MutationStrategy) for candidates and
//! never builds a model call itself.

pub mod config;
pub mod continuous;
pub mod iterative;
pub mod promotion;
pub mod state;

pub use config::SingleHarnessConfig;
pub use continuous::{ContinuousResult, ContinuousSingleHarness};
pub use iterative::{IterativeResult, IterativeSingleHarness};
pub use promotion::{
    batch_filter, case_scores, full_evaluation_decision, is_batch_improvement, mean_improvement,
    stage_one_gate, targets_improved, threshold_saturated, CaseScores, FullEvaluationDecision,
    StageOneDecision, EPSILON, REASON_BATCH_GATE_PASSED, REASON_CONTENT_UNCHANGED,
    REASON_ENGINEERING_FAILED, REASON_PROTECTED_CASE_REGRESSED, REASON_SELECTED,
    REASON_SELECTED_SCORE_REGRESSED, REASON_TARGET_REPLAY_FAILED, REASON_THRESHOLD_SATURATED,
};
pub use state::{
    attempt_key, batch_key, state_path, CandidateGateRecord, JournalEntry, PromotionPolicyMode,
    SingleHarnessState, SingleHarnessStatus, UsageState, STATE_FILE_NAME, STATE_VERSION,
};
